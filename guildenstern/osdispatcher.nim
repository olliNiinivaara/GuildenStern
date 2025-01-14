## TODO: documentation

import net, os, posix, locks, atomics
from nativesockets import accept, setBlocking
from std/osproc import countProcessors
import guildenserver, guildenselectors

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
  except: discard

shutdownCallbacks.add(shutdownImpl)

proc suspend(server: GuildenServer, sleepmillisecs: int) {.gcsafe, nimcall, raises: [].} =
  server.log(TRACE, "suspending thread " & $getThreadId() & " for " & $sleepmillisecs & " millisecs") 
  sleep(sleepmillisecs)
    

{.warning[Deprecated]:off.}
proc closeSocketImpl(server: GuildenServer, socket: posix.SocketHandle, cause: SocketCloseCause, msg: string = "") {.gcsafe, nimcall, raises: [].} =
  try:
    if socket == INVALID_SOCKET:
      server.log(DEBUG, "cannot close invalid socket " & $cause & ": " & msg)
      return
    server.log(TRACE, "osdispatcher now closing socket " & $socket)
    if not isNil(server.onclosesocketcallback):
      server.onclosesocketcallback(server, socket, cause, msg)
    elif not isNil(server.deprecatedOnclosesocketcallback):
      let fakeClient = guildenserver.SocketData(server: server, socket: socket)
      server.deprecatedOnclosesocketcallback(addr fakeClient, cause, msg)
  finally:
    discard posix.close(socket)
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
        event = servers[server.id].clientselector.selectFast()

        if unlikely(Event.User in event.events): break 
        if unlikely(Event.Error in event.events):
          closeSocketImpl(server, SocketHandle(event.fd), NetErrored, "socket " & $event.fd & ": error " & $event.errorCode)
          continue
        socket = SocketHandle(event.fd)
        if unlikely(socket.int in [0, INVALID_SOCKET.int]): continue
        client = getSafelyData(emptyClient, servers[server.id].clientselector, socket.int)
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
    withLock(servers[server.id].flagLock):
      return getSafelyData(emptyClient, servers[server.id].clientselector, socket.int).flags


proc setFlagsImpl(server: GuildenServer, socket: SocketHandle, flags: int): bool {.nimcall.} =
  {.gcsafe.}:
    withLock(servers[server.id].flagLock):
      let client = getSafelyData(emptyClient, servers[server.id].clientselector, socket.int)
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
  {.gcsafe.}:
    server.log(INFO, "osdispatcher " & $server.id & " now listening at port " & $server.port &
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
    servers[server.id].serversocket.close()
    let waitingtime = 10 # 10 seconds, TODO: make this configurable / larger than socket timeout?
    server.log(DEBUG, "Stopping client threads...")
    var slept = 0
    while slept <= 1000 * waitingtime:
      sleep(200)
      slept += 200
      for i in 1 .. servers[server.id].threadpoolsize:
        try: trigger(shutdownevent)
        except: discard
      if servers[server.id].threadpoolsize < 1: break
      if slept == 200:
        server.log(NOTICE, "Waiting for threads to stop...")
    servers[server.id].flaglock.deinitLock()
    if slept > 1000 * waitingtime:
      server.log(NOTICE, "Not all threads stopped after waiting " & $waitingtime & " seconds. Proceeding with shutdown anyway.")
    else: server.log(DEBUG, "threads stopped.\n")
    sleep(200) # wait for OS


proc start*(server: GuildenServer, port: int, threadpoolsize: uint = 0): bool =
  ## Starts the server.thread loop, which then listens the given port for read requests until shuttingdown == true.
  ## By default threadpoolsize will be set to 2 * processor core count.
  doAssert(server.id < MaxServerCount)
  doAssert(not server.started)
  doAssert(not isNil(server.handlerCallback))
  servers[server.id] = Server()
  servers[server.id].flaglock.initLock()
  servers[server.id].threadpoolsize = threadpoolsize.int
  if servers[server.id].threadpoolsize == 0: servers[server.id].threadpoolsize = 2 * countProcessors()
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
    server.log(INFO, "osdispatcher client-server " & $server.id & " now serving, using " & $servers[server.id].threadpoolsize & " threads")
  return not shuttingdown


proc registerSocket*(server: GuildenServer, socket: SocketHandle,  flags = 0,customdata: pointer = nil): bool =
  try:
    let client = new Client
    client.flags = flags
    client.customdata = customdata
    {.gcsafe.}: servers[server.id].clientselector.registerEPOLLETReadHandle(socket.int, client)
    server.log(DEBUG, "socket " & $socket & " connected to client-server " & $server.id)
    return true
  except:
    server.log(ERROR, "registerSocket: selector registerHandle error")
    return false