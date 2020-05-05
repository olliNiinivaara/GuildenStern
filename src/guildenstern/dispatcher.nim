# (c) Copyright 2020 Olli Niinivaara

import guildenserver

import selectors, net, nativesockets, os, httpcore, posix, streams
import private/[httpin, wsin]
from wsout import wsHandshake


when compileOption("threads"):
  import weave
  import locks # because getThreadId not in 0 ..< WEAVE_NUM_THREADS

const
  WEAVE_NUM_THREADS {.intdefine.} = 1
  LOADBALANCE_FREQUENCY {.intdefine.} = 20


template initData(kind: FdKind): Data = Data(fdKind: kind, clientid: NullHandle.Clientid)

template processHttp() =
  if readFromHttp(c):
    try:
      if c.gs.serverstate == Maintenance and c.gs.errorHandler != nil:
        c.gs.errorHandler(c, "")
      else:
        c.gs.httpHandler(c)
        if c.currentexceptionmsg != "":
          if c.gs.errorHandler != nil: c.gs.errorHandler(c, "processHttp: " & c.currentexceptionmsg)
          return        
        if c.clientid.int32 != NullHandle:
          if wsHandshake(c):
            data.fdKind = Ws
            data.clientid = c.clientid
        if c.currentexceptionmsg != "" and
          c.gs.errorHandler != nil: c.gs.errorHandler(c, c.currentexceptionmsg)
    except:
      c.currentexceptionmsg = getCurrentExceptionMsg()
      if c.gs.errorHandler != nil: c.gs.errorHandler(c, "processHttp: " & c.currentexceptionmsg)
    finally:
      if c.fd.int32 != NullHandle and c.gs.selector.contains(c.fd):
        try: c.gs.selector.updateHandle(c.fd, {Event.Read})
        except:
          c.currentexceptionmsg = getCurrentExceptionMsg()
          if c.gs.errorHandler != nil:
            c.gs.errorHandler(c, "updateHandle: " & c.currentexceptionmsg)
  else:
    if c.gs.errorHandler != nil and c.currentexceptionmsg != "": c.gs.errorHandler(c, c.currentexceptionmsg)
    c.gs.closeFd(fd)


proc disconnectWs(c: GuildenVars, closedbyclient: bool) =
  c.gs.closeFd(c.fd)
  {.gcsafe.}:
    if c.gs.wsdisconnectHandler != nil: c.gs.wsdisconnectHandler(c, closedbyclient)


template processWs() =
  let opcode = readFromWs(c)
  if opcode != Opcode.Text:
    if c.currentexceptionmsg != "" and c.gs.errorHandler != nil: c.gs.errorHandler(c, c.currentexceptionmsg)
    c.disconnectWs(opcode == Opcode.Close)
  else:
    try:      
      if c.gs.serverstate == Maintenance and c.gs.errorHandler != nil: c.gs.errorHandler(c, "")
      else:
        c.gs.wsHandler(c)
        if c.currentexceptionmsg != "" and
          c.gs.errorHandler != nil: c.gs.errorHandler(c, c.currentexceptionmsg)
        if c.fd.int32 != NullHandle and c.gs.selector.contains(c.fd):
          try: c.gs.selector.updateHandle(c.fd, {Event.Read})
          except:
            c.currentexceptionmsg = getCurrentExceptionMsg()
            if c.gs.errorHandler != nil:
              c.gs.errorHandler(c, "updateHandle: " & c.currentexceptionmsg)
    except:
      if c.gs.errorHandler != nil: c.gs.errorHandler(c, "processWs: " & getCurrentexceptionMsg())    
      c.disconnectWs(false)

proc process[T: GuildenVars](gs: ptr GuildenServer, threadcontexts: ptr array[WEAVE_NUM_THREADS, T], fd: posix.SocketHandle, data: ptr Data) {.gcsafe, raises: [].} =
  if gs.serverstate == Shuttingdown: return
  {.gcsafe.}:
    var t = 0
    when compileOption("threads"):
      let id = weave.getThreadid()
      while t < WEAVE_NUM_THREADS:
        if threadcontexts[t].threadid == id: break
        t.inc
      if(unlikely) t == WEAVE_NUM_THREADS:
        acquire(gs.ctxlock)
        t = 0
        while t < WEAVE_NUM_THREADS:
          if threadcontexts[t].threadid == -1: (threadcontexts[t].threadid = id ; break)
          t.inc
        release(gs.ctxlock)
        assert(t != WEAVE_NUM_THREADS)
    template c: untyped = threadcontexts[t]
    c.currentexceptionmsg.setLen(0)
    c.path = 0
    c.pathlen = 0
    c.methlen = 0
    c.bodystartpos = 0
    try:
      c.sendbuffer.setPosition(0)
      c.recvbuffer.setPosition(0)
    except: (echo "Nim internal error"; return)
    c.fd = fd
    c.clientid = data.clientid
  discard gs.inflight.atomicinc
  try:
    if data.fdKind == Http: processHttp() else: processWs()
  finally: discard gs.inflight.atomicinc(-1)


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
    try:
      spawn process[T](unsafeAddr gs, unsafeAddr threadcontexts, fd, data)
    except: break
  else: process[T](unsafeAddr gs, unsafeAddr threadcontexts, fd, data)


proc initContext(context: GuildenVars, gs: GuildenServer) =
  context.gs = gs
  context.sendbuffer = newStringStream()
  context.sendbuffer.data.setLen(MaxResponseLength)
  context.wsheader = newStringStream()
  context.wsheader.data.setLen(20)
  context.recvbuffer = newStringStream()
  context.recvbuffer.data.setLen(MaxResponseLength)


proc eventLoop[T: GuildenVars](gs: GuildenServer) {.gcsafe, raises: [].} =
  var eventbuffer: array[1, ReadyKey]

  var threadcontexts: array[WEAVE_NUM_THREADS, T]
  for t in 0 ..<  WEAVE_NUM_THREADS:
    threadcontexts[t] = new T
    threadcontexts[t].initContext()
    initContext(threadcontexts[t], gs)

  while true:
    var ret: int
    {.push assertions: on.} # otherwise selectInto could panic # TODO: minimal repro & raise Nim-lang issue?
    var backoff: int
    var speeding: int
    try:
      if gs.turbo: backoff = 0
      else:
        if gs.inflight < 0: # TODO: remove when sure that never triggered
          echo "flight went below ground zero!"
          gs.inflight = 0
        backoff = if gs.inflight < 1: LOADBALANCE_FREQUENCY else: 0
      when defined(fulldebug):
        if gs.inflight > 1: echo "inflight: ", gs.inflight
      ret = gs.selector.selectInto(backoff, eventbuffer)
      if gs.serverstate == Shuttingdown: break
      if ret != 1:
        when compileOption("threads"):
          try: loadBalance(Weave)
          except: discard
          if not gs.turbo: # TODO: remove when sure that never triggered
            speeding.inc
            if speeding > 10000:
              echo "caught speeding!"
              speeding = 0
              gs.inflight = 0 
        continue
    except: continue
    speeding = 0
    if eventbuffer[0].events.len == 0:
      when compileOption("threads"):
        try: syncRoot(Weave)
        except: discard
      continue
    {.pop.}

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
    except:
      echo "selector.getData: " & getCurrentExceptionMsg()
      break

    if Event.Error in event.events:
      case data.fdKind
      of Server: echo "server error: " & osErrorMsg(event.errorCode)
      of Http:
        if event.errorCode.cint != ECONNRESET: echo "http error: " & osErrorMsg(event.errorCode)
        gs.closeFd(fd)
      of Ws: echo "ws error: " & osErrorMsg(event.errorCode)
      else: discard
      continue

    if Event.Read notin event.events:
      try:
        if gs.selector.contains(fd): gs.selector.unregister(fd)
        nativesockets.close(fd)
      except: discard
      finally: continue
    handleEvent() 


proc serve*[T: GuildenVars](gs: GuildenServer, port: int) =
  doAssert(gs.httpHandler != nil, "No http handler registered")
  when compileOption("threads"):
    doAssert(defined(threadsafe), "Selectors module requires compiling with -d:threadsafe")
    # doAssert(defined(WEAVE_NUM_THREADS), "Weave module requires compiling with -d:WEAVE_NUM_THREADS=n")
    #when defined(fulldebug):
    echo "WEAVE_NUM_THREADS: ", WEAVE_NUM_THREADS
    initLock(gs.ctxlock)
    init(Weave)
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
  when compileOption("threads"):
    syncRoot(Weave)
  gs.serverstate = Normal
  eventLoop[T](gs)
  gs.serverstate = ShuttingDown
  when compileOption("threads"):
    exit(Weave)
    release gs.ctxlock
  echo ""      
  {.gcsafe.}:
    if gs.shutdownHandler != nil: gs.shutdownHandler()
  quit(0)