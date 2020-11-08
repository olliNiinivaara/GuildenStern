import guildenserver

import selectors, net, nativesockets, os, httpcore, posix

when compileOption("threads"): import threadpool


var threadinitialized {.threadvar.}: bool


proc process(gs: ptr GuildenServer, fd: posix.SocketHandle, data: ptr SocketData) {.gcsafe, raises: [].} =
  if shuttingdown: return
  if data.ctxid == InvalidCtx: return
  if not threadinitialized:
    threadinitialized = true
    if gs.threadinitializer != nil: gs.threadinitializer()
  data.socket = fd
  handleRead(gs, data)
  if gs.multithreading:
    if gs.selector.contains(fd):
      try: gs.selector.updateHandle(fd, {Event.Read})
      except: echo "addHandle error: " & getCurrentExceptionMsg()
    else: fd.close()


template handleAccept(theport: uint16) =
  var porthandler = 0
  while gs.porthandlers[porthandler].port != theport.int: porthandler += 1
  let fd = fd.accept()[0]
  if fd == osInvalidSocket: return
  if gs.selector.contains(fd):
    if gs.selector.setData(fd, SocketData(port: gs.porthandlers[porthandler].port.uint16, ctxid: gs.porthandlers[porthandler].ctxid)):
      gs.selector.updateHandle(fd, {Event.Read})
      when defined(fulldebug): echo "socket reconnected: ", fd
    else:
      gs.selector.unregister(fd)
      echo "socket reconnection fail at port ", theport
      return
  else:
    gs.selector.registerHandle(fd, {Event.Read}, SocketData(port: gs.porthandlers[porthandler].port.uint16, ctxid: gs.porthandlers[porthandler].ctxid))
    when defined(fulldebug): echo "socket connected: ", fd
  var tv = (RcvTimeOut,0)
  if setsockopt(fd, cint(SOL_SOCKET), cint(RcvTimeOut), addr(tv), SockLen(sizeof(tv))) < 0'i32:
    gs.selector.unregister(fd)
    raise newException(CatchableError, osErrorMsg(event.errorCode))
  

template handleEvent() =
  if data.ctxid == ServerCtx:
    try:
      handleAccept(data.port)
    except:
      if osLastError().int != 2 and osLastError().int != 9: echo "connect error: " & getCurrentExceptionMsg()
    continue
  if not gs.multithreading: process(unsafeAddr gs, fd, data)
  else:
    try: gs.selector.updateHandle(fd, {})
    except:
      echo "removeHandle error: " & getCurrentExceptionMsg()
      continue
    when compileOption("threads"): spawn process(unsafeAddr gs, fd, data)


proc processTimer(gs: ptr GuildenServer, data: ptr SocketData) =
  if not threadinitialized:
    threadinitialized = true
    if gs.threadinitializer != nil: gs.threadinitializer()
  try: cast[TimerCallback](data.customdata)()
  except: discard


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
        data = addr(gs.selector.getData(fd))
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
          if event.errorCode.cint != 2 and event.errorCode.cint != 9 and event.errorCode.cint != 32 and event.errorCode.cint != 104: echo "socket error: ", event.errorCode.cint," ", osErrorMsg(event.errorCode)
          if gs.selector.contains(fd):
            try: gs.selector.unregister(fd)
            except: discard
          fd.close()
          when defined(fulldebug): echo "socket closed by client: ", fd
        continue

      if Event.Read notin event.events:
        try:
          when defined(fulldebug): echo "non-read ", fd, ": ", event.events
          fd.close()
          if gs.selector.contains(fd): gs.selector.unregister(fd)          
        except: discard
        finally: continue       
      handleEvent()
    except:
      echo "dispatcher: ", getCurrentExceptionMsg()
      writeStackTrace()
      continue


proc serve*(gs: GuildenServer, multithreaded = true) {.gcsafe, nimcall.} =
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
      portserver.setSockOpt(OptReuseAddr, true)
      portserver.setSockOpt(OptReusePort, true)
      try:
        portserver.bindAddr(net.Port(gs.porthandlers[i].port), "")
      except:
        echo "Could not open port ", gs.porthandlers[i].port
        raise
      gs.selector.registerHandle(portserver.getFd(), {Event.Read}, SocketData(port: gs.porthandlers[i].port.uint16, ctxid: ServerCtx))
      portserver.listen()
      portserver.getFd().setBlocking(false)

  {.gcsafe.}: signal(SIG_PIPE, SIG_IGN)
  
  eventLoop(gs)
  for portserver in portservers: portserver.close()