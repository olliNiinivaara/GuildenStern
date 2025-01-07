## This is the default dispatcher that ships with GuildenStern. Use it by importing guildenstern/dispatcher and
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
from nativesockets import accept, setBlocking
from std/osproc import countProcessors
import guildenserver, selector

static:
  when not defined(nimdoc): doAssert(defined(threadsafe))

const
  MaxServerCount {.intdefine.} = 30
  ## Maximum number of servers

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

var servers: array[MaxServerCount, Server]
var emptyClient = Client(flags: -1)


proc suspend(server: GuildenServer, sleepmillisecs: int) {.gcsafe, nimcall, raises: [].} =
    server.log(TRACE, "suspending thread " & $getThreadId() & " for " & $sleepmillisecs & " millisecs") 
    sleep(sleepmillisecs)
    

{.warning[Deprecated]:off.}
proc closeSocketImpl(server: GuildenServer, socket: posix.SocketHandle, cause: SocketCloseCause, msg: string = "") {.gcsafe, nimcall, raises: [].} =
  try:
    if socket == INVALID_SOCKET:
      server.log(DEBUG, "cannot close invalid socket " & $cause & ": " & msg)
      return
    server.log(DEBUG, "Closing socket " & $socket & ": " & $cause & " " & msg)
    if not isNil(server.onclosesocketcallback):
      server.onclosesocketcallback(server, socket, cause, msg)
    elif not isNil(server.deprecatedOnclosesocketcallback):
      let fakeClient = guildenserver.SocketData(server: server, socket: socket)
      server.deprecatedOnclosesocketcallback(addr fakeClient, cause, msg)
  finally:
    discard posix.close(socket)
{.warning[Deprecated]:on.}


template error(s: string): untyped = 
  server.log(TRACE, s)
  continue

proc clientThread(server: GuildenServer) {.thread.} =
  var
    event: ReadyKey
    socket: SocketHandle
    client: Client
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
  
    server.log(TRACE, "handleRead starts for socket " & $socket)
    handleRead(server, socket, client.customdata)
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
      var client = getSafelyData(emptyClient, servers[server.id].clientselector, socket.int)
      if client == emptyClient: return false
      client.flags = client.flags or flags
      return true


proc serverThread(server: GuildenServer) {.thread, gcsafe, nimcall, raises: [].} =
  when not defined(nimdoc):
    var linger = TLinger(l_onoff: 1, l_linger: 0)
    signal(SIG_PIPE, SIG_IGN)

  var serversocket: Socket
  try:
    serversocket = newSocket()
    serversocket.bindAddr(net.Port(server.port), "")
    when not defined(nimdoc):
      discard setsockopt(serversocket.getFd(), cint(SOL_SOCKET), cint(SO_LINGER), addr linger, SockLen(sizeof(TLinger)))
      serversocket.setSockOpt(OptNoDelay, true, level = cint(Protocol.IPPROTO_TCP))
    serversocket.listen()
  except CatchableError, Defect:
    server.log(FATAL, "Could not open port " & $server.port)
    server.started = true
    server.port = 0
    return
  
  {.gcsafe.}:
    servers[server.id].serversocket = serversocket
    var eventbuffer: array[1, ReadyKey]
    try:
      servers[server.id].serverselector = newSelector[bool]()
      servers[server.id].serverselector.registerEvent(shutdownevent, true)
      servers[server.id].serverselector.registerHandle(serversocket.getFd(), {Event.Read}, true)
      servers[server.id].clientselector = newSelector[Client]()
      servers[server.id].clientselector.registerEvent(shutdownevent, Client())
    except:
      server.log(FATAL, "Could not start server selector")
      server.started = true
      server.port = 0
      return

    var clientthreads = newSeq[Thread[GuildenServer]](servers[server.id].threadpoolsize)
    try:
      for i in 0 ..< servers[server.id].threadpoolsize: createThread(clientthreads[i], clientThread, server)
    except ResourceExhaustedError:
      server.log(FATAL, "Could not create worker threads")
      server.started = true
      server.port = 0
      return  

  server.started = true
  server.log(INFO, "Guildenserver with id " & $server.id & " now listening at port " & $server.port & " with socket " & $serversocket.getFd())
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
      var client = new Client
      {.gcsafe.}:
        servers[server.id].clientselector.registerExclusiveReadHandle(newsocket.int, client)
    except:
      server.log(ERROR, "selector registerHandle error for socket " & $newsocket)
      continue
    server.log(DEBUG, "New socket " & $newsocket & " connected at port " & $server.port)

  
  {.gcsafe.}:
    servers[server.id].serversocket.close()
    let waitingtime = 10 # 10 seconds, TODO: make this configurable / larger than socket timeout?
    server.log(DEBUG, "Waiting for threads to stop...")
    var slept = 0
    while slept <= 1000 * waitingtime:
      sleep(100)
      slept += 100
      for i in 1 .. servers[server.id].threadpoolsize:
        try: trigger(shutdownevent)
        except: discard
      if servers[server.id].threadpoolsize == 0: break
    servers[server.id].flaglock.deinitLock()
    if slept > 1000 * waitingtime:
      server.log(NOTICE, "Not all threads stopped after waiting " & $waitingtime & " seconds. Proceeding with shutdown anyway.")
    else: server.log(DEBUG, "threads stopped.\n")
    sleep(200) # wait for OS

proc start*(server: GuildenServer, port: int, threadpoolsize: uint = 0): bool {.discardable.} =
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
  createThread(server.thread, serverThread, server)
  while not server.started:
    sleep(50)
    if shuttingdown: break
  if server.port == 0: shuttingdown = true
  return not shuttingdown


proc start*(server: GuildenServer, port: int, threadpoolsize: uint = 0, maxactivethreadcount: uint): bool {.discardable, deprecated:"maxactivethreadcount is not used".} =
   return start(server, port, threadpoolsize)