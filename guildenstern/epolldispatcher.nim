## This is an epoll-based dispatcher for platforms that support it, like Linux.
##
## Servers are started by using the [start] proc.
##
## This dispatcher delegates the thread scheduling to the operating system kernel,
## making it a rock solid dispatcher alternative. If your platform supports it,
## it is the recommended choice.

import net, os, posix, locks, atomics
from nativesockets import accept, setBlocking
from std/osproc import countProcessors
import guildenserver
when not defined(nimdoc): import guildenselectors
else: import std/selectors

const
  MSG_DONTWAIT = when defined(macosx): 0x80.cint else: 0x40.cint
  MaxServerCount {.intdefine.} = 30
  ## Maximum number of servers for this dispatcher

type
  Client = ref object
    flags: int
    customdata: pointer
    processing: Atomic[bool]

  Server = object
    serversocket: Socket
    serverselector: Selector[bool]
    clientselector: Selector[Client]
    flaglock: Lock
    threadpoolsize: int
    clientthreads: seq[Thread[GuildenServer]]

var
  servers: array[MaxServerCount, Server]
  emptyClient = Client(flags: -1)
  shutdownevent = newSelectEvent()


proc shutdownImpl() {.nimcall, gcsafe, raises: [].} =
  try: trigger(shutdownevent)
  except: echo getCurrentExceptionMsg()

shutdownCallbacks.add(shutdownImpl)

proc suspend(server: GuildenServer, sleepmillisecs: int) {.gcsafe, nimcall, raises: [].} =
  server.log(TRACE, "suspending thread " & $getThreadId() & " for " & $sleepmillisecs & " millisecs") 
  sleep(sleepmillisecs)
    

{.warning[Deprecated]:off.}
proc closeSocketImpl(server: GuildenServer, socket: posix.SocketHandle, cause: SocketCloseCause, msg: string = "") {.gcsafe, nimcall, raises: [].} =
  if socket == INVALID_SOCKET:
    server.log(DEBUG, "cannot close invalid socket " & $cause & ": " & msg)
    return
  server.log(TRACE, "epolldispatcher now closing socket " & $socket)
  if not isNil(server.onclosesocketcallback):
    server.onclosesocketcallback(server, socket, cause, msg)
  elif not isNil(server.deprecatedOnclosesocketcallback):
    let fakeClient = guildenserver.SocketData(server: server, socket: socket)
    server.deprecatedOnclosesocketcallback(addr fakeClient, cause, msg)
  
  when not defined(nimdoc):
    {.gcsafe.}:
      if servers[server.id].clientselector.contains(socket):
        try:
          servers[server.id].clientselector.unregister(socket.int)
          discard posix.close(socket)
        except:
          server.log(TRACE, "error unregistering socket " & $socket)
{.warning[Deprecated]:on.}


proc clientThread(server: GuildenServer) {.thread.} =
  var
    event: ReadyKey
    socket: SocketHandle
    client: Client
    probebuffer = newString(1)

  initializeThread(server)
  while true:
    if unlikely(shuttingdown): break
    try:
      {.gcsafe.}:
        when not defined(nimdoc): event = servers[server.id].clientselector.selectFast()

        if unlikely(Event.User in event.events): break 
        if unlikely(Event.Error in event.events):
          closeSocketImpl(server, SocketHandle(event.fd), NetErrored, "socket " & $event.fd & ": error " & $event.errorCode)
          continue
        socket = SocketHandle(event.fd)
        if unlikely(socket.int in [0, INVALID_SOCKET.int]): continue
        when not defined(nimdoc): client = getSafelyData(emptyClient, servers[server.id].clientselector, socket.int)
        if unlikely(client.flags == -1): continue
    except:
      server.log(INFO, "client selector select failure")
      continue
    if unlikely(shuttingdown): break

    let alreadybeingprocessed = exchange(client.processing, true)
    if alreadybeingprocessed == true: continue
  
    server.log(TRACE, "handleRead starts for socket " & $socket & " at thread " & $getThreadId())

    while true:
      handleRead(server, socket, client.customdata)
      if unlikely(shuttingdown): break
      let ret = recv(socket, addr probebuffer[0], 1, MSG_PEEK or MSG_DONTWAIT)
      if ret < 0:
        let state = osLastError().cint
        if state == EAGAIN: break
        elif state == EWOULDBLOCK: continue
        else: break
      elif ret == 0: break
      else: continue

    store(client.processing, false)
    server.log(TRACE, "handleRead finished for socket " & $socket)

  if not isNil(server.threadFinalizerCallback): server.threadFinalizerCallback()
  {.gcsafe.}: discard servers[server.id].threadpoolsize.atomicDec()


proc getFlagsImpl(server: GuildenServer, socket: SocketHandle): int {.nimcall.} =
  {.gcsafe.}:
    when not defined(nimdoc):
      withLock(servers[server.id].flagLock):
        return getSafelyData(emptyClient, servers[server.id].clientselector, socket.int).flags
    else: discard


proc setFlagsImpl(server: GuildenServer, socket: SocketHandle, flags: int): bool {.nimcall.} =
  {.gcsafe.}:
    withLock(servers[server.id].flagLock):
      when not defined(nimdoc):
        let client = getSafelyData(emptyClient, servers[server.id].clientselector, socket.int)
      else:
        let client = emptyClient
      if client == emptyClient: return false
      client.flags = client.flags or flags
      server.log(DEBUG, "Socket " & $socket & " flags set to " & $flags)
      return true


proc createSelector(server: GuildenServer): bool =
  when not defined(nimdoc):
    var linger = TLinger(l_onoff: 1, l_linger: 0)
    signal(SIG_PIPE, SIG_IGN)

  try:
    servers[server.id].serversocket = newSocket()
    servers[server.id].serversocket.bindAddr(net.Port(server.port), "")
    when not defined(nimdoc):
      discard setsockopt(servers[server.id].serversocket.getFd(), cint(SOL_SOCKET), cint(SO_LINGER), addr linger, SockLen(sizeof(TLinger)))
      servers[server.id].serversocket.setSockOpt(OptNoDelay, true, level = cint(Protocol.IPPROTO_TCP))
    servers[server.id].serversocket.listen()
  except:
    server.log(FATAL, "Could not open port " & $server.port)
    return false
  
  {.gcsafe.}:
    try:
      servers[server.id].serverselector = newSelector[bool]()
      servers[server.id].serverselector.registerEvent(shutdownevent, true)
      servers[server.id].serverselector.registerHandle(servers[server.id].serversocket.getFd(), {Event.Read}, true)
      return true
    except:
      server.log(FATAL, "Could not create selectors for port " & $server.port)
      return false


proc startClientthreads(server: GuildenServer): bool =
  try:
    servers[server.id].clientselector = newSelector[Client]()
    servers[server.id].clientselector.registerEvent(shutdownevent, Client())
    for i in 0 ..< servers[server.id].threadpoolsize:
      servers[server.id].clientthreads.add(Thread[GuildenServer]())
      createThread(servers[server.id].clientthreads[i], clientThread, server)
    return true
  except:
    server.log(FATAL, "Could not create client threads")
    return false


proc listeningLoop(server: GuildenServer) {.thread, gcsafe, nimcall, raises: [].} =
  var eventbuffer: array[1, ReadyKey]
  when not defined(nimdoc):
    {.gcsafe.}:
      server.log(INFO, "epolldispatcher " & $server.id & " now listening at port " & $server.port &
      " with socket " & $servers[server.id].serversocket.getFd() & " using " & $servers[server.id].threadpoolsize & " threads")
  server.started = true
  while true:
    try:
      {.gcsafe.}: discard servers[server.id].serverselector.selectInto(-1, eventbuffer)
    except:
      server.log(FATAL, "server select failed")

    if unlikely(shuttingdown): break

    let event = eventbuffer[0]

    if unlikely(event.events.len == 0):
      server.log(TRACE, "no events in event")
      continue

    if unlikely(Event.Signal in event.events):
      server.log(INFO, "Signal event detected...")
      continue

    if unlikely(Event.Process in event.events):
      server.log(INFO, "Process event detected...")
      continue
    
    if unlikely(Event.Error in event.events):
      server.log(ERROR, "server selector thread error: " & osErrorMsg(event.errorCode))
      continue
 
    if unlikely(Event.Read notin event.events):
      server.log(TRACE, "skipping event: " & $event.events)
      continue

    let fd = posix.SocketHandle(event.fd)
    server.log(TRACE, "Serversocket: " & $event.events)

    when not defined(nimdoc):
      let newsocket = fd.accept()[0]
      if unlikely(newsocket.int in [0, INVALID_SOCKET.int]):
        server.log(TRACE, "invalid new socket")
        continue
      try:
        let client = new Client
        {.gcsafe.}:
          servers[server.id].clientselector.registerEPOLLETReadHandle(newsocket.int, client)
      except:
        server.log(ERROR, "selector registerHandle error for socket " & $newsocket)
        continue
      server.log(DEBUG, "New socket " & $newsocket & " connected at port " & $server.port)

  
  {.gcsafe.}:
    when not defined(nimdoc):
      try: servers[server.id].serversocket.close()
      except: server.log(ERROR, "could not close serversocket " & $servers[server.id].serversocket.getFd())
    let waitingtime = 10 # 10 seconds, TODO: make this configurable / larger than socket timeout?
    server.log(DEBUG, "Stopping client threads...")
    var slept = 0
    while slept <= 1000 * waitingtime:
      sleep(200)
      slept += 200
      try: trigger(shutdownevent)
      except: echo getCurrentExceptionMsg()
      if servers[server.id].threadpoolsize < 1: break
      if slept == 200: server.log(INFO, "waiting for threads to stop...")
      server.log(TRACE, "threads still running: " & $servers[server.id].threadpoolsize)
    servers[server.id].flaglock.deinitLock()
    if slept > 1000 * waitingtime:
      server.log(NOTICE, "Not all threads stopped after waiting " & $waitingtime & " seconds. Proceeding with shutdown anyway.")
    else: server.log(DEBUG, "threads stopped.\n")
    sleep(200) # wait for OS


proc start*(server: GuildenServer, port: int, threadpoolsize: uint = 0): bool =
  ## Starts the server.thread loop, which then listens the given port for read requests until shuttingdown == true.
  ## Threadpoolsize means number of worker threads, but the name is kept for compatibility with the defautl dispatacher.
  ## By default threadpoolsize will be set to max(8, 2 * countProcessors()).
  ## If port number 0 is given, the server runs in client mode, where server.thread is not started, and sockets 
  ## can be added manually using the `registerSocket` proc.
  doAssert(server.id < MaxServerCount)
  doAssert(not server.started)
  doAssert(not isNil(server.handlerCallback))
  servers[server.id] = Server()
  servers[server.id].flaglock.initLock()
  servers[server.id].threadpoolsize = threadpoolsize.int
  if servers[server.id].threadpoolsize == 0: servers[server.id].threadpoolsize = max(8, 2 * countProcessors())
  server.port = port.uint16
  server.suspendCallback = suspend
  server.closeSocketCallback = closeSocketImpl
  server.getFlagsCallback = getFlagsImpl
  server.setFlagsCallback = setFlagsImpl
  if port > 0 and not createSelector(server): return false
  if not startClientthreads(server): return false
  if port > 0:
    createThread(server.thread, listeningLoop, server)
    while not server.started:
      sleep(50)
      if shuttingdown: return false
  sleep(200) # wait for OS
  if port == 0:
    server.log(INFO, "epolldispatcher client-server " & $server.id & " now serving, using " & $servers[server.id].threadpoolsize & " threads")
  return not shuttingdown


proc registerSocket*(server: GuildenServer, socket: SocketHandle,  flags = 0,customdata: pointer = nil): bool =
  ## Add a socket whose read events will be then dispatched. Useful for servers operating in client mode.
  try:
    let client = new Client
    client.flags = flags
    client.customdata = customdata
    when not defined(nimdoc):
      {.gcsafe.}: servers[server.id].clientselector.registerEPOLLETReadHandle(socket.int, client)
    server.log(DEBUG, "socket " & $socket & " registered to server " & $server.id)
    return true
  except:
    server.log(ERROR, "registerSocket: selector registerHandle error")
    return false