import guildenserver

import selectors, net, os, httpcore, posix, locks
from nativesockets import accept, setBlocking
from osproc import countProcessors

include threadqueue

var
  workerthreadscreated = 0

proc handleAccept(gs: ptr GuildenServer, fd: posix.SocketHandle, data: ptr SocketData) =
  when not defined(nimdoc):
    let fd = fd.accept()[0]
  if fd == INVALID_SOCKET: return

  if data.socket == fd and gs.selector.contains(fd.int):
    gs.selector.updateHandle(fd.int, {Event.Read})
    gs[].log(INFO, "socket reconnected: " & $fd)
    return

  if gs.selector.contains(fd.int):
    try: gs.selector.unregister(fd.int)
    except CatchableError, Defect:
      gs[].log(WARN, "could not unregister contained socket: " & $fd)
      return

  var ctxid = 0
  while gs.porthandlers[ctxid.CtxId].port != data.port: ctxid += 1
  gs.selector.registerHandle(fd.int, {Event.Read},
   SocketData(port: gs.porthandlers[ctxid.CtxId].port, ctxid: ctxid.CtxId, socket: fd))

  gs[].log(DEBUG, "socket connected: " & $fd)
  var tv = (RcvTimeOut,0)
  if setsockopt(fd, cint(posix.SOL_SOCKET), cint(RcvTimeOut), addr(tv), SockLen(sizeof(tv))) < 0'i32:
    gs.selector.unregister(fd.int)
    gs[].log(ERROR, "setting timeout failed: " & $fd)



template handleEvent() =
  if unlikely(data.ctxid == ServerCtx):
    try:
      handleAccept(unsafeAddr gs, fd, data)
    except CatchableError, Defect:
      if osLastError().int != 2 and osLastError().int != 9: gs.log(ERROR, "connect error")
      else:
        gs.log(INFO, "connect error")
    continue

  {.gcsafe.}:
    if unlikely(gs.workerthreadcount == 1):
        data.socket = fd
        handleRead(unsafeAddr gs, data)
    else:
      if likely(data.port > 0):
        try: gs.selector.updateHandle(fd.int, {})
        except CatchableError, Defect:
          gs.log(ERROR, "removeHandle error: " & getCurrentExceptionMsg())
          continue
      createTask()

  
proc eventLoop(gs: GuildenServer) {.gcsafe, raises: [].} =
  var eventbuffer: array[1, ReadyKey]
  while true:
    try:
      var ret: int
      try: ret = gs.selector.selectInto(-1, eventbuffer)
      except CatchableError, Defect:
        gs.log(ERROR, "selector.select")
        continue

      gs.log(TRACE, "event")
      if unlikely(shuttingdown): break
      if unlikely(eventbuffer[0].events.len == 0):
        discard sched_yield()
        continue
      
      let event = eventbuffer[0]
      if unlikely(Event.Signal in event.events):
        gs.log(INFO, "Signal event detected...")
        continue
      let fd = posix.SocketHandle(event.fd)
      gs.log(TRACE, "socket " & $fd & ": " & $event.events)
      var data: ptr SocketData
      try:
        {.push warning[ProveInit]: off.}
        data = addr(gs.selector.getData(fd.int))
        {.pop.}
        if unlikely(data == nil): continue
      except CatchableError, Defect:
        gs.log(FATAL, "selector.getData error")
        break

      if unlikely(Event.Timer in event.events):
        handleEvent()
        continue

      if unlikely(Event.Error in event.events):
        if data.ctxid == ServerCtx: gs.log(ERROR, "server error: " & osErrorMsg(event.errorCode))
        else:
          data.socket = fd
          let cause =
            if event.errorCode.cint in [2,9]: AlreadyClosed
            elif event.errorCode.cint == 32: ConnectionLost
            elif event.errorCode.cint == 104: ClosedbyClient
            else: NetErrored 
          closeOtherSocket(unsafeAddr gs, data, cause, osErrorMsg(event.errorCode))
        continue

      if unlikely(Event.Read notin event.events):
        try:
          gs.log(INFO, "non-read " & $fd & ": " & $event.events)
          if gs.workerthreadcount == 1: closeOtherSocket(unsafeAddr gs, data, NetErrored, "non-read " & $fd & ": " & $event.events)
          else:
            when compileOption("threads"): closeOtherSocket(unsafeAddr gs, data, NetErrored, "non-read " & $fd & ": " & $event.events)
        except CatchableError, Defect: discard
        finally: continue

      handleEvent()
    except CatchableError, Defect:
      gs.log(FATAL, "dispatcher exception")
      continue


{.push hints:off.}
proc serve*(gs: GuildenServer, threadcount = -1, loglevel: LogLevel = ERROR) {.gcsafe, nimcall.} =
  gs.loglevel = loglevel
  var linger = TLinger(l_onoff: 1, l_linger: 0)

  if gs.selector == nil: (gs.log(FATAL, "no handlers registered"); quit())
  if threadcount != 1 and not compileOption("threads"): gs.workerthreadcount = 1
  elif threadcount < 1: gs.workerthreadcount = countProcessors() + 2
  else: gs.workerthreadcount = threadcount

  gs.log(NOTICE, "gs.workerthreadcount: " & $gs.workerthreadcount)

  {.gcsafe.}:
    if gs.workerthreadcount == 1:
      if threadinitializer != nil: threadinitializer()
    else:
      doAssert(defined(threadsafe), "When threads:on, selectors module requires compiling with -d:threadsafe") 
      if workerthreadscreated == 0:
        initCond(workavailable)
        initLock(worklock)
      while workerthreadscreated < gs.workerthreadcount:        
        when compileOption("threads"):
          var thread: Thread[void]
          createThread(thread, threadProc)
          sleep(5)
        workerthreadscreated += 1

    var portservers: seq[Socket]
    for i in 0 .. MaxCtxHandlers:
      if gs.porthandlers[i.CtxId].port > 0:
        let port = gs.porthandlers[i.CtxId].port
        let portserver = newSocket()
        portservers.add(portserver)
        try:
          discard setsockopt(posix.SocketHandle(portserver.getFd()), cint(SOL_SOCKET), cint(SO_LINGER), addr linger, SockLen(sizeof(TLinger)))
          portserver.bindAddr(net.Port(port), "")
        except CatchableError, Defect:
          gs.log(FATAL, "Could not open port " & $port)
          raise
        gs.selector.registerHandle(portserver.getFd().int, {Event.Read}, SocketData(port: port, ctxid: ServerCtx))
        portserver.listen()
        portserver.getFd().setBlocking(false)

    signal(SIG_PIPE, SIG_IGN)

  sleep(10)
  eventLoop(gs)

  gs.log(NOTICE, "guildenstern dispatcher loop stopped")
  for portserver in portservers: portserver.close()
{.pop.}