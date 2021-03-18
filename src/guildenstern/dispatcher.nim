import guildenserver

import selectors, net, os, httpcore, posix
from nativesockets import accept, setBlocking

when compileOption("threads"): import threadpool

var currentload, peekload, maxload: int
var threadinitialized {.threadvar.}: bool


proc getLoads*(): (int, int, int) = (currentload, peekload, maxload)


proc process(gs: ptr GuildenServer, fd: posix.SocketHandle, data: ptr SocketData) {.gcsafe, raises: [].} =
  if shuttingdown: return
  if data.ctxid == InvalidCtx: return
  if not threadinitialized:
    threadinitialized = true
    if gs.threadinitializer != nil: gs.threadinitializer()
  data.socket = fd
  handleRead(gs, data)
  if gs.multithreading:
    currentload.atomicDec()
    if currentload == 0: peekload = 0
    if gs.selector.contains(fd.int):
      try: gs.selector.updateHandle(fd.int, {Event.Read})
      except: echo "addHandle error: " & getCurrentExceptionMsg()


proc handleAccept(gs: ptr GuildenServer, fd: posix.SocketHandle, data: ptr SocketData) =
  when not defined(nimdoc):
    let fd = fd.accept()[0]
  if fd == INVALID_SOCKET: return

  if data.socket == fd and gs.selector.contains(fd.int):
    gs.selector.updateHandle(fd.int, {Event.Read})
    when defined(fulldebug): echo "socket reconnected: ", fd
    return

  if gs.selector.contains(fd.int):
    try: gs.selector.unregister(fd.int)
    except:
      when defined(fulldebug): echo "could not unregister contained socket: ", fd
      return

  var porthandler = 0
  while gs.porthandlers[porthandler].port != data.port.int: porthandler += 1
  gs.selector.registerHandle(fd.int, {Event.Read},
   SocketData(port: gs.porthandlers[porthandler].port.uint16, ctxid: gs.porthandlers[porthandler].ctxid, protocol: gs.porthandlers[porthandler].protocol, socket: fd))

  when defined(fulldebug): echo "socket connected: ", fd
  var tv = (RcvTimeOut,0)
  if setsockopt(fd, cint(posix.SOL_SOCKET), cint(RcvTimeOut), addr(tv), SockLen(sizeof(tv))) < 0'i32:
    gs.selector.unregister(fd.int)
    when defined(fulldebug): echo "setting timeout failed: ", fd


template handleEvent() =
  if data.ctxid == ServerCtx:
    try:
      handleAccept(unsafeAddr gs, fd, data)
    except:
      if osLastError().int != 2 and osLastError().int != 9: echo "connect error: " & getCurrentExceptionMsg()
      else:
        if defined(fulldebug): echo "connect error: " & getCurrentExceptionMsg()
    continue

  if not gs.multithreading: process(unsafeAddr gs, fd, data)
  else:
    currentload.atomicInc()
    if currentload > peekload:
      peekload = currentload
      if peekload > maxload:
        maxload = peekload
    try: gs.selector.updateHandle(fd.int, {})
    except:
      echo "removeHandle error: " & getCurrentExceptionMsg()
      continue
    when compileOption("threads"): spawn process(unsafeAddr gs, fd, data)


proc processTimer(gs: ptr GuildenServer, data: ptr SocketData) =
  if not threadinitialized:
    threadinitialized = true
    if gs.threadinitializer != nil: gs.threadinitializer()
  {.gcsafe.}:  
    try: cast[TimerCallback](data.customdata)()
    except:
      if defined(fulldebug): echo "timer: " & getCurrentExceptionMsg()


template handleTimer() =
  if not gs.multithreading: processTimer(unsafeAddr gs, data)
  else:
    when compileOption("threads"): spawn processTimer(unsafeAddr gs, data)
  

proc eventLoop(gs: GuildenServer) {.gcsafe, raises: [].} =
  var eventbuffer: array[1, ReadyKey]
  while true:
    try:
      var ret: int
      try: ret = gs.selector.selectInto(-1, eventbuffer)
      except: discard
      when defined(fulldebug): echo "event"    
      if shuttingdown: break
      
      let event = eventbuffer[0]
      if Event.Signal in event.events:
        echo "Signal event detected..."
        continue
      let fd = posix.SocketHandle(event.fd)
      if event.events.len == 0: continue
      when defined(fulldebug): echo "socket ", fd, ": ", event.events
      var data: ptr SocketData
      try:
        {.push warning[ProveInit]: off.}
        data = addr(gs.selector.getData(fd.int))
        {.pop.}
        if data == nil: continue
      except:
        echo "selector.getData error: " & getCurrentExceptionMsg()
        break
      if Event.Timer in event.events:
        handleTimer()
        continue
      if Event.Error in event.events:
        if data.ctxid == ServerCtx: echo "server error: " & osErrorMsg(event.errorCode)
        else:
          data.socket = fd
          let cause =
            if event.errorCode.cint in [2,9]: AlreadyClosed
            elif event.errorCode.cint == 32: ConnectionLost
            elif event.errorCode.cint == 104: ClosedbyClient
            else: NetErrored 
          if not gs.multithreading: closeOtherSocket(unsafeAddr gs, data, cause, osErrorMsg(event.errorCode))
          else:
            when compileOption("threads"): spawn closeOtherSocket(unsafeAddr gs, data, cause, osErrorMsg(event.errorCode))
        continue

      if Event.Read notin event.events:
        try:
          when defined(fulldebug): echo "non-read ", fd, ": ", event.events
          if not gs.multithreading: closeOtherSocket(unsafeAddr gs, data, NetErrored, "non-read " & $fd & ": " & $event.events)
          else:
            when compileOption("threads"): closeOtherSocket(unsafeAddr gs, data, NetErrored, "non-read " & $fd & ": " & $event.events)
        except: discard
        finally: continue       
      handleEvent()
    except:
      echo "dispatcher: ", getCurrentExceptionMsg()
      writeStackTrace()
      continue


proc serve*(gs: GuildenServer, multithreaded = true) {.gcsafe, nimcall.} =  
  var linger = TLinger(l_onoff: 1, l_linger: 0)

  if gs.selector == nil: (echo "no handlers registered"; quit())
  gs.multithreading = multithreaded
  if gs.multithreading and not compileOption("threads"):
    echo "threads:off; serving single-threaded"
    gs.multithreading = false
  if gs.multithreading: doAssert(defined(threadsafe), "Selectors module requires compiling with -d:threadsafe")
  var portservers: seq[Socket]
  {.gcsafe.}:
    for i in 0 ..< gs.portcount:
      let portserver = newSocket()      
      portservers.add(portserver)
      try:
        discard setsockopt(posix.SocketHandle(portserver.getFd()), cint(SOL_SOCKET), cint(SO_LINGER), addr linger, SockLen(sizeof(TLinger)))
        portserver.bindAddr(net.Port(gs.porthandlers[i].port), "")
      except:
        echo "Could not open port ", gs.porthandlers[i].port
        raise
      gs.selector.registerHandle(portserver.getFd().int, {Event.Read}, SocketData(port: gs.porthandlers[i].port.uint16, ctxid: ServerCtx))
      portserver.listen()
      portserver.getFd().setBlocking(false)

  {.gcsafe.}: signal(SIG_PIPE, SIG_IGN)
  
  sleep(10)
  eventLoop(gs)

  when defined(fulldebug): echo "guildenstern dispatcher loop stopped"
  for portserver in portservers: portserver.close()