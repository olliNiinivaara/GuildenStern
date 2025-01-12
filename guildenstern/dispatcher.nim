## This is the default dispatcher. Use it by importing guildenstern/dispatcher and
## then starting a server's dispatcher thread by calling [start].
## 
## This dispatcher spawns a thread from a server-specific threadpool for 
## every incoming read request. The threadpool size can be set in the start procedure.
## Dispatcher avoids excessive thread swapping by not spawning a thread when
## activethreadcount has reached maxactivethreadcount.
## However, if a thread reports itself as inactive (by calling server's suspend procedure),
## another thread is allowed to run. This keeps the server serving when there are
## more threads suspended than the maxactivethreadcount. This design seems to deliver decent
## performance without leaning to more complex asynchronous concurrency techniques.    

import net, os, posix, locks
import selectors
from nativesockets import accept, setBlocking
from std/osproc import countProcessors
import guildenserver

static:
  when not defined(nimdoc): doAssert(defined(threadsafe), "This dispatcher requires the -d:threadsade compiler switch")

const
  QueueSize {.intdefine.} = 200
  ## Empirically found number. Should be larger than maxactivethreadcount.
  MaxServerCount {.intdefine.} = 30
  ## Maximum number of servers


type
  SocketData = object
    socket: SocketHandle = INVALID_SOCKET
    isserversocket: bool
    flags: int
    customdata: pointer

  WorkerData = object
    gsselector: Selector[SocketData]
    queue: array[QueueSize, ptr SocketData]
    tail: int
    head: int = -1
    workavailable: Cond
    worklock: Lock
    activethreadcount: int
    maxactivethreadcount: int
    threadpoolsize: int

var
  shutdownevent = newSelectEvent()
  workerdatas: array[MaxServerCount, WorkerData]

template wd(): untyped = workerdatas[server.id]

proc shutdownImpl() {.nimcall, gcsafe, raises: [].} =
  try: trigger(shutdownevent)
  except: discard

shutdownCallbacks.add(shutdownImpl)

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
  initializeThread(server)
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
    handleRead(server, wd.queue[mytail].socket, wd.queue[mytail].customdata)
    server.log(TRACE, "handled event at queue position " & $mytail)

    restoreRead(server, wd.gsselector, wd.queue[mytail].socket.int)
  if not isNil(server.threadFinalizerCallback): server.threadFinalizerCallback()


#[proc findSocketDataForSocket(server: GuildenServer, socket: posix.SocketHandle): ptr SocketData =
  {.gcsafe.}:
    for socketdata in workerdatas[server.id].queue: 
      if socketdata.socket == socket: return socketdata
    return addr NoSocketdata]#

proc findSelectorForSocket(server: GuildenServer, socket: posix.SocketHandle): Selector[SocketData] =
  {.gcsafe.}:
    if isNil(workerdatas[server.id].gsselector): return nil
    if not workerdatas[server.id].gsselector.contains(socket.int): return nil
    return workerdatas[server.id].gsselector

{.warning[Deprecated]:off.}
proc closeSocketImpl(server: GuildenServer, socket: posix.SocketHandle, cause: SocketCloseCause, msg: string = "") {.gcsafe, nimcall, raises: [].} =
  if socket == INVALID_SOCKET:
    server.log(DEBUG, "cannot close invalid socket " & $cause & ": " & msg)
    return
  if not isNil(server.onclosesocketcallback):
    server.onclosesocketcallback(server, socket, cause, msg)
  elif not isNil(server.deprecatedOnclosesocketcallback):
    let fakesocketdata = guildenserver.SocketData(server: server, socket: socket)
    server.deprecatedOnclosesocketcallback(addr fakesocketdata, cause, msg)
  
  let theselector = findSelectorForSocket(server, socket)
  try:
    if not isNil(theselector): theselector.unregister(socket.int)
  except:
    server.log(ERROR, "error unregistering socket")
  discard posix.close(socket)
{.warning[Deprecated]:on.}


proc getFlags(server: GuildenServer, socket: posix.SocketHandle): int =
  for socketdata in workerdatas[server.id].queue.mitems(): 
    if socketdata.socket == socket: return socketdata.flags
  return -1
  

proc setFlags(server: GuildenServer, socket: posix.SocketHandle, flags: int): bool =
  for socketdata in workerdatas[server.id].queue.mitems(): 
    if socketdata.socket == socket:
      socketdata.flags = flags
      server.log(DEBUG, "Socket " & $socket & " flags set to " & $flags)
      return true 
  return false


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
    socketdata = addr wd.gsselector.getData(fd.int)
    {.pop.}
  except:
    server.log(FATAL, "selector.getData error")
    return

  socketdata.socket = fd

  if unlikely(Event.Error in event.events):
    if socketdata.isserversocket: server.log(ERROR, "server error: " & osErrorMsg(event.errorCode))
    else:
      let cause =
        if event.errorCode.cint in [2,9]: AlreadyClosed
        elif event.errorCode.cint == 14: EFault
        elif event.errorCode.cint == 32: ConnectionLost
        elif event.errorCode.cint == 104: ClosedbyClient
        else: NetErrored
      closeSocketImpl(server, fd, cause, osErrorMsg(event.errorCode))
    return

  if unlikely(Event.Read notin event.events):
    try:
      server.log(INFO, "dysunctional " & $fd & ": " & $event.events)
      closeSocketImpl(server, fd, NetErrored, "non-read " & $fd & ": " & $event.events)
    except: discard
    finally: return

  if unlikely(socketdata.isserversocket):
    try:
      when not defined(nimdoc):
        let fd = fd.accept()[0]
      if unlikely(fd.int in [0, INVALID_SOCKET.int]):
        server.log(TRACE, "invalid new socket")
        return
      wd.gsselector.registerHandle(fd.int, {Event.Read}, SocketData(isserversocket: false, socket: fd))
      server.log(DEBUG, "socket " & $fd & " connected to thread " & $getThreadId())
    except:
      server.log(ERROR, "selector registerHandle error")
    finally:
      return

  if wd.activethreadcount > wd.maxactivethreadcount:
    sleep(0)
    return

  try:
    wd.gsselector.updateHandle(fd.int, {})
  except:
    server.log(ERROR, "remove read handle error")
    return

  if unlikely(wd.head == QueueSize - 1):
    server.log(TRACE, "queue head reached the end, stalling for " & $(wd.head - wd.tail) & " requests")
    while likely(wd.tail < QueueSize - 1):
      if wd.activethreadcount < wd.maxactivethreadcount:
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
  {.gcsafe.}:
    let gsselector = workerdatas[server.id].gsselector
    server.log(INFO, "dispatcher now listening at port " & $server.port) 
  server.started = true
  

  while true:
    if unlikely(shuttingdown): break
    var ret: int
    try: ret = gsselector.selectInto(-1, eventbuffer)
    except:
      server.log(ERROR, "selector.select")
      continue
    if unlikely(server.loglevel == TRACE):
      eventid += 1
      server.log(TRACE, "\L--- event nr " & $eventid & " received ---")
    if unlikely(shuttingdown): break
    processEvent(server, eventbuffer[0])


proc createSelector(server: GuildenServer): bool =
  try:
    {.gcsafe.}:
      signal(SIG_PIPE, SIG_IGN)
      workerdatas[server.id].gsselector = newSelector[SocketData]()
      workerdatas[server.id].gsselector.registerEvent(shutdownevent, SocketData())
  except:
    server.log(FATAL, "Could not create selector")
    return false
  return true


proc startListening(server: GuildenServer): bool =
  when not defined(nimdoc):
    var linger = TLinger(l_onoff: 1, l_linger: 0)
  var portserver: Socket
  try:
    portserver = newSocket()
    portserver.bindAddr(net.Port(server.port), "")
    when not defined(nimdoc):
      discard setsockopt(portserver.getFd(), cint(SOL_SOCKET), cint(SO_LINGER), addr linger, SockLen(sizeof(TLinger)))
    portserver.listen()
  except:
    server.log(FATAL, "Could not open port " & $server.port)
    return false
  try:
    {.gcsafe.}:
      workerdatas[server.id].gsselector.registerHandle(portserver.getFd().int, {Event.Read}, SocketData(isserversocket: true))
    when not defined(nimdoc): portserver.getFd().setBlocking(false)
    portserver.setSockOpt(OptNoDelay, true, level = cint(Protocol.IPPROTO_TCP))
  except:
    server.log(FATAL, "Could not listen to port " & $server.port)
    return false
  return true
  

proc startEventloop(server: GuildenServer) {.thread, gcsafe, nimcall, raises: [].} =
  {.gcsafe.}:
    var workerthreads = newSeq[Thread[GuildenServer]](workerdatas[server.id].threadpoolsize)
    try:
      for i in 0 ..< workerdatas[server.id].threadpoolsize: createThread(workerthreads[i], workerthreadLoop, server)
    except ResourceExhaustedError:
      server.log(FATAL, "Could not create worker threads")
      server.started = true
      server.port = 1
      return

  eventLoop(server)

  try:
    trigger(shutdownevent)  
    {.gcsafe.}:
      for i in 0 ..< workerdatas[server.id].threadpoolsize: signal(workerdatas[server.id].workavailable) 
    sleep(50)
    trigger(shutdownevent)
  except:
    server.log(FATAL, "shutdown failed")

  let waitingtime = 10 # 10 seconds, TODO: make this configurable / larger than socket timeout
  server.log(DEBUG, "waiting for client threads to stop...")
  var slept = 0
  while slept < 1000 * waitingtime:
    sleep(500)
    slept += 500
    for t in workerthreads:
      if t.running: continue
    break
  if slept > 1000 * waitingtime:
    server.log(NOTICE, "Not all threads stopped after waiting " & $waitingtime & " seconds. Proceeding with shutdown anyway.")
  else: server.log(DEBUG, "threads stopped.")
  sleep(200) # wait for OS


proc start*(server: GuildenServer, port: int, threadpoolsize: uint = 0, maxactivethreadcount: uint = 0): bool =
  ## Starts the server.thread loop, which then listens the given port for read requests until shuttingdown == true.
  ## By default maxactivethreadcount will be set to processor core count, and threadpoolsize twice that.
  ## If you a running lots of servers, or if this server is not under a heavy load, these numbers can be lowered.
  doAssert(server.id < MaxServerCount)
  doAssert(not server.started)
  doAssert(port != 1)
  doAssert(threadpoolsize >= maxactivethreadcount)
  workerdatas[server.id] = WorkerData()
  workerdatas[server.id].head = -1
  initCond(workerdatas[server.id].workavailable)
  initLock(workerdatas[server.id].worklock)
  workerdatas[server.id].maxactivethreadcount = maxactivethreadcount.int
  if workerdatas[server.id].maxactivethreadcount == 0: workerdatas[server.id].maxactivethreadcount = countProcessors()
  workerdatas[server.id].threadpoolsize = threadpoolsize.int
  if workerdatas[server.id].threadpoolsize == 0: workerdatas[server.id].threadpoolsize = workerdatas[server.id].maxactivethreadcount * 2
  server.port = port.uint16
  server.suspendCallback = suspend
  server.closeSocketCallback = closeSocketImpl
  server.getFlagsCallback = getFlags
  server.setFlagsCallback = setFlags
  if not createSelector(server): return false
  if server.port > 0 and not startListening(server): return false
  createThread(server.thread, startEventloop, server)
  while not server.started:
    sleep(50)
    if shuttingdown: return false
  return server.port != 1 and not shuttingdown


proc registerSocket*(theserver: GuildenServer, socket: SocketHandle, customdata: pointer = nil): bool =
  try:
    workerdatas[theserver.id].gsselector.registerHandle(socket.int, {Event.Read}, SocketData(isserversocket: false, socket: socket, customdata: customdata))
    theserver.log(DEBUG, "socket " & $socket & " connected to client-server " & $theserver.id)
    return true
  except:
    theserver.log(ERROR, "registerSocket: selector registerHandle error")
    return false