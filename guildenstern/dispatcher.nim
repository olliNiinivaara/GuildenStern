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

import net, os, posix, locks, atomics
from strutils import startsWith
from nativesockets import accept, setBlocking
from std/osproc import countProcessors
import guildenserver, guildenselectors


const
  QueueSize {.intdefine.} = 200
  ## Empirically found number. Should be larger than maxactivethreadcount.
  MaxServerCount {.intdefine.} = 30
  ## Maximum number of servers


type
  SocketData = ref object 
    socket: SocketHandle = INVALID_SOCKET
    isserversocket: bool
    flags: int
    customdata: pointer
    isprocessing: Atomic[bool]

  WorkerData = object
    gsselector: Selector[SocketData]
    queue: array[QueueSize, SocketData]
    tail: int
    head: int = -1
    workavailable: Cond
    worklock: Lock
    flaglock: Lock
    activethreadcount: int
    maxactivethreadcount: int
    threadpoolsize: int

proc `=copy`(dest: var WorkerData; source: WorkerData) {.error.}

var
  shutdownevent = newSelectEvent()
  workerdatas: array[MaxServerCount, WorkerData]
  emptySocketData = SocketData(flags: -1)


proc shutdownImpl() {.nimcall, gcsafe, raises: [].} =
  try: trigger(shutdownevent)
  except: discard

shutdownCallbacks.add(shutdownImpl)


proc suspend(server: GuildenServer, sleepmillisecs: int) {.gcsafe, nimcall, raises: [].} =
  {.gcsafe.}:
    discard workerdatas[server.id].activethreadcount.atomicdec()
    server.log(TRACE, "suspending thread " & $getThreadId() & " for " & $sleepmillisecs & " millisecs") 
    sleep(sleepmillisecs)
    discard workerdatas[server.id].activethreadcount.atomicinc()


proc getSelectorForSocket(server: GuildenServer, socket: posix.SocketHandle): Selector[SocketData] =
  {.gcsafe.}:
    if isNil(workerdatas[server.id].gsselector): return nil
    if not workerdatas[server.id].gsselector.contains(socket.int): return nil
    return workerdatas[server.id].gsselector


{.warning[Deprecated]:off.}
proc closeSocketImpl(server: GuildenServer, socket: posix.SocketHandle, cause: SocketCloseCause, msg: string = "") {.gcsafe, nimcall, raises: [].} =
  if unlikely(shuttingdown): return 
  if socket == INVALID_SOCKET:
    server.log(DEBUG, "cannot close invalid socket " & $cause & ": " & msg)
    return
  if not isNil(server.onclosesocketcallback):
    server.onclosesocketcallback(server, socket, cause, msg)
  elif not isNil(server.deprecatedOnclosesocketcallback):
    let fakesocketdata = guildenserver.SocketData(server: server, socket: socket)
    server.deprecatedOnclosesocketcallback(addr fakesocketdata, cause, msg)
  
  let theselector = getSelectorForSocket(server, socket)
  try:
    if not isNil(theselector): theselector.unregister(socket.int)
  except:
    server.log(TRACE, "error unregistering socket " & $socket)
  discard posix.close(socket)
{.warning[Deprecated]:on.}


proc restoreRead(server: GuildenServer, selector: Selector[SocketData], socketdata: SocketData) {.inline.} =
  if unlikely(shuttingdown or socketdata.socket == INVALID_SOCKET) or not selector.contains(socketdata.socket): return
  try:
    selector.updateHandle(socketdata.socket.int, {Event.Read})
  except:
    if getCurrentExceptionMsg().startsWith("File exists"):
      server.log(WARN, "Selector tried to restore existing read")
    else:
      closeSocket(server, socketdata.socket, NetErrored, "Could not restore Read event to selector")
  socketdata.isprocessing.store(false)


proc workerthreadLoop(server: GuildenServer) {.thread.} =
  {.gcsafe.}:
    initializeThread(server)
    workerdatas[server.id].activethreadcount.atomicInc()
    while true:
      if unlikely(shuttingdown): break
      if workerdatas[server.id].tail >= workerdatas[server.id].head:  
        workerdatas[server.id].activethreadcount.atomicDec()
        withLock(workerdatas[server.id].worklock):
          wait(workerdatas[server.id].workavailable, workerdatas[server.id].worklock)
        workerdatas[server.id].activethreadcount.atomicInc()
        continue
      let mytail = workerdatas[server.id].tail.atomicInc()
      if unlikely(mytail > workerdatas[server.id].head): continue

      if unlikely(shuttingdown): break

      server.log(TRACE, "handling event at queue position " & $mytail)
      handleRead(server, workerdatas[server.id].queue[mytail].socket, workerdatas[server.id].queue[mytail].customdata)
      server.log(TRACE, "handled event at queue position " & $mytail)

      restoreRead(server, workerdatas[server.id].gsselector, workerdatas[server.id].queue[mytail])
  if not isNil(server.threadFinalizerCallback): server.threadFinalizerCallback()


proc getFlagsImpl(server: GuildenServer, socket: SocketHandle): int {.nimcall, raises: [].} =
  {.gcsafe.}:
    withLock(workerdatas[server.id].flagLock):
      let selector = server.getSelectorForSocket(socket)
      if not selector.isNil: return getSafelyData(emptySocketData, selector, socket.int).flags
      else: return -1


proc setFlagsImpl(server: GuildenServer, socket: SocketHandle, flags: int): bool {.nimcall.} =
  {.gcsafe.}:
    withLock(workerdatas[server.id].flagLock):
      let selector = server.getSelectorForSocket(socket)
      if selector.isNil: return false
      var client = getSafelyData(emptySocketData, selector, socket.int)
      if client == emptySocketData: return false
      client.flags = client.flags or flags
      server.log(DEBUG, "Socket " & $socket & " flags set to " & $flags)
      return true


proc processEvent(server: GuildenServer, event: ReadyKey) {.gcsafe, raises: [].} =
  {.gcsafe.}:
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

    var socketdata = getSafelyData(emptySocketData, workerdatas[server.id].gsselector, fd.int)
    if unlikely(socketdata == emptySocketData):
      closeSocket(server, fd, NetErrored, "selected socket disappeared")
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
          server.log(DEBUG, "invalid new socket")
          return
        workerdatas[server.id].gsselector.registerHandle(fd.int, {Event.Read}, SocketData(isserversocket: false, socket: fd))
        server.log(DEBUG, "socket " & $fd & " connected to thread " & $getThreadId())
      except:
        server.log(ERROR, "selector registerHandle error")
      finally:
        return

    if workerdatas[server.id].activethreadcount > workerdatas[server.id].maxactivethreadcount:
      sleep(0)
      return

    let wasprocessing = exchange(socketdata.isprocessing, true)
    if wasprocessing: return
    try:
      workerdatas[server.id].gsselector.updateHandle(fd.int, {})
    except:
      socketdata.isprocessing.store(false)
      server.log(ERROR, "remove read handle error")
      return

    if unlikely(workerdatas[server.id].head == QueueSize - 1):
      server.log(TRACE, "queue head reached the end, stalling for " & $(workerdatas[server.id].head - workerdatas[server.id].tail) & " requests")
      while likely(workerdatas[server.id].tail < QueueSize - 1):
        if workerdatas[server.id].activethreadcount < workerdatas[server.id].maxactivethreadcount:
          signal(workerdatas[server.id].workavailable)
        discard sched_yield()
        if unlikely(shuttingdown): break
      workerdatas[server.id].tail = 0
      workerdatas[server.id].head = -1
      
    if unlikely(shuttingdown): return
    
    workerdatas[server.id].queue[workerdatas[server.id].head + 1] = socketdata
    if workerdatas[server.id].tail > workerdatas[server.id].head: workerdatas[server.id].tail = workerdatas[server.id].head
    workerdatas[server.id].head.inc()
    signal(workerdatas[server.id].workavailable)


proc eventLoop(server: GuildenServer) {.gcsafe, raises: [].} =
  var event: ReadyKey
  var eventid: int
  {.gcsafe.}:
    let gsselector = workerdatas[server.id].gsselector
    if server.port > 0: server.log(INFO, "dispatcher " & $server.id & " now listening at port " & $server.port &
    " using " & $workerdatas[server.id].threadpoolsize & " threads")
    else: server.log(INFO, "dispatcher client-server " & $server.id & " now serving, using " & $workerdatas[server.id].threadpoolsize & " threads")
  server.started = true
  
  while true:
    if unlikely(shuttingdown): break
    try: event = gsselector.selectFast()
    except:
      server.log(ERROR, "selector.select")
      continue
    if unlikely(server.loglevel <= DEBUG):
      eventid += 1
      server.log(DEBUG, "\L--- event "  & $eventid & " received ---")
    if unlikely(shuttingdown): break
    processEvent(server, event)


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
  server.log(DEBUG, "Stopping client threads...")
  var slept = 0
  var stillworking = false

  while slept < 1000 * waitingtime:
    stillworking = false
    sleep(200)
    slept += 200
    for t in workerthreads:
      if t.running:
        stillworking = true
        break
    if stillworking:
      if slept == 200:
        {.gcsafe.}:
          server.log(NOTICE, "waiting for threads to stop")
      try: trigger(shutdownevent)
      except: discard
    else: break

  if slept > 1000 * waitingtime:
    server.log(NOTICE, "not all threads stopped after waiting " & $waitingtime & " seconds. Proceeding with shutdown anyway.")
  else: server.log(DEBUG, "threads stopped.")
  {.gcsafe.}:
    deinitLock(workerdatas[server.id].flaglock)
    deinitLock(workerdatas[server.id].worklock)
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
  initLock(workerdatas[server.id].flaglock)
  workerdatas[server.id].maxactivethreadcount = maxactivethreadcount.int
  if workerdatas[server.id].maxactivethreadcount == 0: workerdatas[server.id].maxactivethreadcount = countProcessors()
  workerdatas[server.id].threadpoolsize = threadpoolsize.int
  if workerdatas[server.id].threadpoolsize == 0: workerdatas[server.id].threadpoolsize = workerdatas[server.id].maxactivethreadcount * 2
  server.port = port.uint16
  server.suspendCallback = suspend
  server.closeSocketCallback = closeSocketImpl
  server.getFlagsCallback = getFlagsImpl
  server.setFlagsCallback = setFlagsImpl
  if not createSelector(server): return false
  if server.port > 0 and not startListening(server): return false
  createThread(server.thread, startEventloop, server)
  while not server.started:
    sleep(50)
    if shuttingdown: return false
  if server.port != 1 and not shuttingdown:
    sleep(200) # wait for OS
    return true
  else: return false
 

proc registerSocket*(theserver: GuildenServer, socket: SocketHandle, flags = 0, customdata: pointer = nil): bool =
  try:
    workerdatas[theserver.id].gsselector.registerHandle(socket.int, {Event.Read}, SocketData(isserversocket: false, socket: socket, flags: flags, customdata: customdata))
    theserver.log(DEBUG, "socket " & $socket & " connected to client-server " & $theserver.id)
    return true
  except:
    theserver.log(ERROR, "registerSocket: selector registerHandle error")
    return false