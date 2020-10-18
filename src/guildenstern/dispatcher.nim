# (c) Copyright 2020 Olli Niinivaara

import guildenserver

import selectors, net, nativesockets, os, httpcore, posix, streams
import private/[httpin, wsin]
from wsout import wsHandshake

when compileOption("threads"):
  import threadpool, locks
  const MAXTHREADCONTEXTSIZE = MaxThreadPoolSize
  var threadreservations: array[MAXTHREADCONTEXTSIZE, bool]
  var lock: Lock

else:
  const MAXTHREADCONTEXTSIZE = 1


template initData(kind: FdKind): Data = Data(fdKind: kind, clientid: NullHandle.Clientid)

template processHttp() =
  if readFromHttp(gv):
    try:
      if gv.gs.serverstate == Maintenance and gv.gs.errorHandler != nil:
        gv.gs.errorHandler(gv, "")
      else:
        gv.gs.httpHandler(gv)
        if gv.currentexceptionmsg != "":
          if gv.gs.errorHandler != nil: gv.gs.errorHandler(gv, "processHttp: " & gv.currentexceptionmsg)
          return        
        if gv.clientid.int32 != NullHandle:
          if wsHandshake(gv):
            data.fdKind = Ws
            data.clientid = gv.clientid
        if gv.currentexceptionmsg != "" and
          gv.gs.errorHandler != nil: gv.gs.errorHandler(gv, gv.currentexceptionmsg)
    except:
      gv.currentexceptionmsg = getCurrentExceptionMsg()
      if gv.gs.errorHandler != nil: gv.gs.errorHandler(gv, "processHttp: " & gv.currentexceptionmsg)
    finally:
      if gv.fd.int32 != NullHandle and gv.gs.selector.contains(gv.fd):
        try: gv.gs.selector.updateHandle(gv.fd, {Event.Read})
        except:
          gv.currentexceptionmsg = getCurrentExceptionMsg()
          if gv.gs.errorHandler != nil:
            gv.gs.errorHandler(gv, "updateHandle: " & gv.currentexceptionmsg)
  else:
    if gv.gs.errorHandler != nil and gv.currentexceptionmsg != "": gv.gs.errorHandler(gv, gv.currentexceptionmsg)
    gv.gs.closeFd(fd)


proc disconnectWs(gv: GuildenVars, closedbyclient: bool) =
  gv.gs.closeFd(gv.fd)
  {.gcsafe.}:
    if gv.gs.wsdisconnectHandler != nil: gv.gs.wsdisconnectHandler(gv, closedbyclient)


template processWs() =
  let opcode = readFromWs(gv)
  if opcode != Opcode.Text:
    if gv.currentexceptionmsg != "" and gv.gs.errorHandler != nil: gv.gs.errorHandler(gv, gv.currentexceptionmsg)
    gv.disconnectWs(opcode == Opcode.Close)
  else:
    try:      
      if gv.gs.serverstate == Maintenance and gv.gs.errorHandler != nil: gv.gs.errorHandler(gv, "")
      else:
        gv.gs.wsHandler(gv)
        if gv.currentexceptionmsg != "" and
          gv.gs.errorHandler != nil: gv.gs.errorHandler(gv, gv.currentexceptionmsg)
        if gv.fd.int32 != NullHandle and gv.gs.selector.contains(gv.fd):
          try: gv.gs.selector.updateHandle(gv.fd, {Event.Read})
          except:
            gv.currentexceptionmsg = getCurrentExceptionMsg()
            if gv.gs.errorHandler != nil:
              gv.gs.errorHandler(gv, "updateHandle: " & gv.currentexceptionmsg)
    except:
      if gv.gs.errorHandler != nil: gv.gs.errorHandler(gv, "processWs: " & getCurrentexceptionMsg())    
      gv.disconnectWs(false)


proc reserveThreadcontext(): int {.inline.} =
  when not compileOption("threads"): return 0
  else:
    withLock(lock):
      var contextid = 0
      while threadreservations[contextid] == true:
        contextid += 1
        if contextid == MAXTHREADCONTEXTSIZE:
          contextid = 0
      threadreservations[contextid] = true
      return contextid


proc releaseThreadcontext(contextid: int) {.inline.} =
  when compileOption("threads"): threadreservations[contextid] = false
  
  
proc process[T: GuildenVars](gs: ptr GuildenServer, threadcontexts: ptr array[MAXTHREADCONTEXTSIZE, T], fd: posix.SocketHandle, data: ptr Data) {.gcsafe, raises: [].} =
  if gs.serverstate == Shuttingdown: return
  let threadcontextid = reserveThreadcontext()
  {.gcsafe.}:
    var gv = threadcontexts[threadcontextid]
  gv.currentexceptionmsg.setLen(0)
  gv.path = 0
  gv.pathlen = 0
  gv.methlen = 0
  gv.bodystartpos = 0
  try:
    gv.sendbuffer.setPosition(0)
    gv.recvbuffer.setPosition(0)
  except: (echo "Nim internal error"; return)
  gv.fd = fd
  gv.clientid = data.clientid
  if data.fdKind == Http: processHttp() else: processWs()
  releaseThreadcontext(threadcontextid)


template handleAccept() =
  let fd = fd.accept()[0]
  if fd == osInvalidSocket: return
  if gs.selector.contains(fd):
    if not gs.selector.setData(fd, initData(Http)): return
  else: gs.selector.registerHandle(fd, {Event.Read}, initData(Http))
  var tv = (RcvTimeOut,0)
  if setsockopt(fd, cint(SOL_SOCKET), cint(RcvTimeOut), addr(tv), SockLen(sizeof(tv))) < 0'i32:
    gs.selector.unregister(fd)
    raise newException(CatchableError, osErrorMsg(event.errorCode))


template handleEvent() =
  if data.fdKind == Server:
    try:
      handleAccept()
    except:
      {.gcsafe.}:
        if osLastError().int != 2 and gs.errorHandler != nil:
          echo "connect error: " & getCurrentExceptionMsg()
    continue
  when compileOption("threads"):
    try: gs.selector.updateHandle(fd, {})
    except:
      echo "updateHandle error: " & getCurrentExceptionMsg()
      continue
    spawn process[T](unsafeAddr gs, unsafeAddr threadcontexts, fd, data)
  else: process[T](unsafeAddr gs, unsafeAddr threadcontexts, fd, data)


proc initGuildenVars(context: GuildenVars, gs: GuildenServer) =
  context.gs = gs
  context.sendbuffer = newStringStream()
  context.sendbuffer.data.setLen(MaxResponseLength)
  context.wsheader = newStringStream()
  context.wsheader.data.setLen(20)
  context.recvbuffer = newStringStream()
  context.recvbuffer.data.setLen(MaxResponseLength)


proc eventLoop[T: GuildenVars](gs: GuildenServer) {.gcsafe, raises: [].} =
  var eventbuffer: array[1, ReadyKey]
  var threadcontexts: array[MAXTHREADCONTEXTSIZE, T]
  for t in 0 ..< MAXTHREADCONTEXTSIZE:
    threadcontexts[t] = new T
    threadcontexts[t].initGuildenVars(gs)
  while true:
    try:
      var ret: int
      try:
        {.push assertions: on.} # otherwise selectInto could panic - a Nim-lang issue
        ret = gs.selector.selectInto(-1, eventbuffer)
        {.pop.}
      except: discard    
      if gs.serverstate == Shuttingdown: break
      if ret != 1 or eventbuffer[0].events.len == 0:
        sleep(0) 
        continue
      
      let event = eventbuffer[0]
      if Event.Signal in event.events: break
      if Event.Timer in event.events:
        {.gcsafe.}: gs.timerHandler()
        continue
      let fd = posix.SocketHandle(event.fd)
      var data: ptr Data        
      try:
        {.push warning[ProveInit]: off.}
        data = addr(gs.selector.getData(fd))
        {.pop.}
        if data == nil: continue
      except:
        echo "selector.getData: " & getCurrentExceptionMsg()
        break

      if Event.Error in event.events:
        case data.fdKind
        of Server: echo "server error: " & osErrorMsg(event.errorCode)
        else:
          if event.errorCode.cint != ECONNRESET: echo "socket error: " & osErrorMsg(event.errorCode)
          gs.closeFd(fd)
        continue

      if Event.Read notin event.events:
        try:
          if gs.selector.contains(fd): gs.selector.unregister(fd)
          nativesockets.close(fd)
        except: discard
        finally: continue
      handleEvent()
    except:
      echo "fail: ", getCurrentExceptionMsg() # TODO: remove if never triggered
      continue
  gs.serverstate = ShuttingDown


proc serve*[T: GuildenVars](gs: GuildenServer, port: int) =
  doAssert(gs.httpHandler != nil, "No http handler registered")
  when compileOption("threads"):
    doAssert(defined(threadsafe), "Selectors module requires compiling with -d:threadsafe")
    initLock(lock)
  let server = newSocket()
  server.setSockOpt(OptReuseAddr, true)
  server.setSockOpt(OptReusePort, true)
  gs.tcpport = net.Port(port)
  try:
    server.bindAddr(gs.tcpport, "")
  except:
    echo "Could not open port ", $gs.tcpport
    raise
  gs.selector = newSelector[Data]()
  gs.selector.registerHandle(server.getFd(), {Event.Read}, initData(Server))
  discard gs.selector.registerSignal(SIGINT, initData(Signal))
  server.listen()
  gs.serverstate = Normal
  eventLoop[T](gs)
  echo ""      
  {.gcsafe.}:
    if gs.shutdownHandler != nil: gs.shutdownHandler()
  quit(0)