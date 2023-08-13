## .. importdoc::  guildenserver.nim

import selectors, net, os, posix, locks
from nativesockets import accept, setBlocking, close
import guildenserver


static: doAssert(defined(threadsafe))

const QueueSize {.intdefine.} = 200

type
  WorkerData = ref object
    gsselector: Selector[SocketData]
    queue: array[QueueSize, ptr SocketData]
    tail: int
    head: int = -1
    workavailable: Cond
    worklock: Lock
    activethreadcount: int
    suspendedthreadcount: int


var
  workerdatas: array[30, WorkerData]
  gsselector {.threadvar.}: Selector[SocketData]
  workerthreads {.threadvar.} : seq[Thread[GuildenServer]]


#[proc getLoad*(server: GuildenServer): int =
  ## returns number of currently running worker threads of this server.
  ## Use server's [maxactivethreadcount] parameter for control.
  return workerdatas[server.id].activethreadcount]#


proc suspend(server: GuildenServer, sleepmillisecs: int) {.gcsafe, nimcall, raises: [].} =
  {.gcsafe.}:
    discard workerdatas[server.id].activethreadcount.atomicdec()
    sleep(sleepmillisecs)
    discard workerdatas[server.id].activethreadcount.atomicinc()


proc restoreRead(server: GuildenServer, selector: Selector[SocketData], socket: int) {.inline.} =
  var success = true
  var failures = 0
  if unlikely(socket != INVALID_SOCKET.int) and likely(selector.contains(socket)):
    try: selector.updateHandle(socket, {Event.Read})
    except: success = false
    while not success:
      sleep(failures)
      try:
        selector.updateHandle(socket, {Event.Read})
        success = true
      except:
        failures += 1
        if failures == 10: success = true
    if unlikely(failures == 10):
      server.log(ERROR, "add read handle error for socket " & $socket)


proc workerthreadLoop(server: GuildenServer) {.thread.} =
  var wd: WorkerData
  {.gcsafe.}:
    if workerdatas[server.id] == nil:
      workerdatas[server.id] = new WorkerData
      workerdatas[server.id].head = -1
      initCond(workerdatas[server.id].workavailable)
      initLock(workerdatas[server.id].worklock)
    wd = workerdatas[server.id]

  wd.activethreadcount.atomicInc()
  while true:
    if unlikely(shuttingdown): break
    if wd.tail >= wd.head:  
      wd.activethreadcount.atomicDec()
      withLock(wd.worklock):
        wait(wd.workavailable, wd.worklock)
      wd.activethreadcount.atomicInc()
      continue
    let mytail = wd.tail.atomicInc()
    if unlikely(mytail > wd.head): continue

    server.log(TRACE, "handling event at queue position " & $mytail)
    handleRead(wd.queue[mytail])
    server.log(TRACE, "handled event at queue position " & $mytail)

    restoreRead(server, wd.gsselector, wd.queue[mytail].socket.int)


proc findSelectorForSocket(server: GuildenServer, socket: posix.SocketHandle): Selector[SocketData] =
  {.gcsafe.}:
    if workerdatas[server.id].gsselector == nil: return nil
    if not workerdatas[server.id].gsselector.contains(socket): return nil
    return workerdatas[server.id].gsselector


proc closeSocketImpl(socketdata: ptr SocketData, cause: SocketCloseCause, msg: string) {.gcsafe, nimcall, raises: [].} =
  if socketdata.socket.int in [0, INVALID_SOCKET.int]:
    socketdata.server.log(TRACE, "use of invalid socket: " & $socketdata.socket)
    return
  socketdata.server.log(DEBUG, "socket " & $cause & ": " & $socketdata.socket & "  " & msg)
  if unlikely(socketdata.isserversocket): socketdata.server.log(DEBUG, "note: closing socket " & $socketdata.socket & " is server socket")
  try:
    if socketdata.server.onclosesocketcallback != nil: socketdata.server.onCloseSocketCallback(socketdata, cause, msg)
    if cause != ClosedbyClient:
      socketdata.socket.close()
      let theselector = findSelectorForSocket(socketdata.server, socketdata.socket)
      if theselector != nil: theselector.unregister(socketdata.socket)
  except Defect, CatchableError:
    socketdata.server.log(ERROR, "socket close error")
  finally:
    socketdata.socket = INVALID_SOCKET


proc closeOtherSocketInOtherThreadImpl(server: GuildenServer, socket: posix.SocketHandle, cause: SocketCloseCause, msg: string = "") {.gcsafe, nimcall, raises: [].} =
  if socket.int in [0, INVALID_SOCKET.int]: return
  server.log(DEBUG, "closeOtherSocketInOtherThread " & $cause & ": " & $socket & "  " & msg)

  if server.onclosesocketcallback != nil:
    var gs = SocketData()
    gs.server = server
    gs.socket = socket
    server.onCloseSocketCallback(addr gs, cause, msg) 
  
  if cause != ClosedbyClient:
    socket.close()
    let theselector = findSelectorForSocket(server, socket)
    try:
      if theselector != nil: theselector.unregister(socket)
    except CatchableError, Defect:
      server.log(ERROR, "error closing socket in another thread")


proc processEvent(server: GuildenServer, event: ReadyKey) {.gcsafe, raises: [].} =
  if unlikely(event.events.len == 0):
    discard sched_yield()
    server.log(TRACE, "no events in event")
    return

  if unlikely(Event.Signal in event.events):
    server.log(INFO, "Signal event detected...")
    return

  if unlikely(Event.Process in event.events):
    server.log(INFO, "Process event detected...")
    return

  let fd = posix.SocketHandle(event.fd)
  server.log(TRACE, "socket " & $fd & ": " & $event.events)

  var socketdata: ptr SocketData
  try:
    {.push warning[ProveInit]: off.}
    socketdata = addr(gsselector.getData(fd.int))
    {.pop.}
    if unlikely(socketdata == nil):
      server.log(TRACE, "socketdata missing")
      return
  except CatchableError, Defect:
    server.log(FATAL, "selector.getData error")
    return

  socketdata.socket = fd

  if unlikely(Event.Error in event.events):
    if socketdata.isserversocket: server.log(ERROR, "server error: " & osErrorMsg(event.errorCode))
    else:
      let cause =
        if event.errorCode.cint in [2,9]: AlreadyClosed
        elif event.errorCode.cint == 32: ConnectionLost
        elif event.errorCode.cint == 104: ClosedbyClient
        else: NetErrored
      closeSocketImpl(socketdata, cause, osErrorMsg(event.errorCode))
    return

  if unlikely(Event.Read notin event.events):
    try:
      server.log(INFO, "dysunctional " & $fd & ": " & $event.events)
      closeSocketImpl(socketdata, NetErrored, "non-read " & $fd & ": " & $event.events)
    except CatchableError, Defect: discard
    finally: return

  if unlikely(socketdata.isserversocket):
    try:
      when not defined(nimdoc):
        let fd = fd.accept()[0]
      if fd.int in [0, INVALID_SOCKET.int]:
        server.log(TRACE, "invalid new socket")
        return
      gsselector.registerHandle(fd.int, {Event.Read}, SocketData(server: server, isserversocket: false, socket: fd))
      server.log(DEBUG, "socket " & $fd & " connected to thread " & $getThreadId())
    except CatchableError, Defect:
      server.log(ERROR, "selector registerHandle error")
    finally:
      return

  var wd: WorkerData
  {.gcsafe.}: wd = workerdatas[server.id]
  if unlikely(wd.gsselector == nil): wd.gsselector = gsselector

  if wd.activethreadcount > server.maxactivethreadcount:
    sleep(0)
    return

  try:
    gsselector.updateHandle(fd.int, {})
  except:
    server.log(ERROR, "remove read handle error")
    return

  if unlikely(wd.head == QueueSize - 1):
    while likely(wd.tail < QueueSize - 1):
      if wd.activethreadcount < server.maxactivethreadcount:
        signal(wd.workavailable)
      discard sched_yield()
      if unlikely(shuttingdown): break
    wd.tail = 0
    wd.head = -1
    
  if unlikely(shuttingdown): return
  
  wd.queue[wd.head + 1] = socketdata
  if wd.tail > wd.head: wd.tail = wd.head
  wd.head.inc()
  signal(wd.workavailable)


proc eventLoop(server: GuildenServer) {.gcsafe, raises: [].} =
  var eventbuffer: array[1, ReadyKey]
  var eventid: int
  server.started = true
  while true:
    if unlikely(shuttingdown): break
    var ret: int
    try: ret = gsselector.selectInto(-1, eventbuffer)
    except CatchableError, Defect:
      server.log(ERROR, "selector.select")
      continue
    if unlikely(server.loglevel == TRACE):
      eventid += 1
      server.log(TRACE, "\L--- event nr " & $eventid & " received ---")
    if unlikely(shuttingdown): break
    processEvent(server, eventbuffer[0])


proc dispatch(serverptr: ptr GuildenServer) {.thread, gcsafe, nimcall, raises: [].} =
  let server = serverptr[]
  server.threadid = getThreadId()
  when not defined nimdoc:
    var linger = TLinger(l_onoff: 1, l_linger: 0)
  try:
    {.gcsafe.}: signal(SIG_PIPE, SIG_IGN)
    gsselector = newSelector[SocketData]()
    gsselector.registerEvent(shutdownevent, SocketData())
  except Exception, CatchableError, Defect:
    server.log(FATAL, "Could not create selector")
    server.started = true
    server.port = 0
    return

  var portserver: Socket
  try:
    portserver = newSocket()
    portserver.bindAddr(net.Port(server.port), "")
    when not defined nimdoc:
      discard setsockopt(portserver.getFd(), cint(SOL_SOCKET), cint(SO_LINGER), addr linger, SockLen(sizeof(TLinger)))
    portserver.listen()
  except CatchableError, Defect:
    server.log(FATAL, "Could not open port " & $server.port)
    server.started = true
    server.port = 0
    return
  try:
    gsselector.registerHandle(portserver.getFd().int, {Event.Read}, SocketData(server: server, isserversocket: true))
    portserver.getFd().setBlocking(false)
    portserver.setSockOpt(OptNoDelay, true, level = cint(Protocol.IPPROTO_TCP))
  except CatchableError, Defect:
    server.log(FATAL, "Could not listen to port " & $server.port)
    server.started = true
    server.port = 0
    return

  workerthreads = newSeq[Thread[GuildenServer]](server.availablethreadcount)
  try:
    for i in 0 ..< server.availablethreadcount: createThread(workerthreads[i], workerthreadLoop, server)
  except ResourceExhaustedError:
    server.log(FATAL, "Could not create worker threads")
    server.started = true
    server.port = 0
    return

  eventLoop(server)

  if shuttingdown:
    try:
      trigger(shutdownevent)
      sleep(50)
      trigger(shutdownevent)
    except:
      server.log(FATAL, "shutdown failed")
      discard


proc start*(server: GuildenServer, port: int, threadcount: uint = 0) =
  doAssert(not server.started)
  var threadcount = threadcount.int
  if threadcount == 0: threadcount = server.maxactivethreadcount * 2
  server.port = port.uint16
  server.suspendCallback = suspend
  server.closeSocketCallback = closeSocketImpl
  server.closeOtherSocketCallback = closeOtherSocketInOtherThreadImpl
  server.availablethreadcount = threadcount.int
  createThread(server.thread, dispatch, unsafeAddr server)
  while not server.started:
    sleep(50)
    if shuttingdown: break
  if server.port == 0: shuttingdown = true