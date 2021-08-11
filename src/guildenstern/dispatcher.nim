import guildenserver

import selectors, net, os, httpcore, posix, locks
from nativesockets import accept, setBlocking
from osproc import countProcessors

var
  workerthreadcount = -1
  workerthreadscreated: bool
  workqueued: bool
  workavailable: Cond
  worklock: Lock
  currentload, peakload, maxload: int
  threadgs: ptr GuildenServer
  threadfd: posix.SocketHandle
  threaddata: ptr SocketData
  threadinitializer: proc() {.nimcall, gcsafe, raises: [].}


proc setWorkerThreadCount*(count: int) =
  workerthreadcount = count


proc registerThreadInitializer*(callback: proc() {.nimcall, gcsafe, raises: [].}) =
  threadinitializer = callback


proc getLoads*(): (int, int, int) = (currentload, peakload, maxload)


proc threadProc() {.thread.} =
  if threadinitializer != nil: threadinitializer()
  while true:
    wait(workavailable, worklock)
    currentload.atomicInc()
    if currentload > peakload:
      peakload = currentload
      if peakload > maxload:
        maxload = peakload
    let
      gs = threadgs
      fd = threadfd
      data = threaddata
    workqueued = false
    release(worklock)
    if data.ctxid == TimerCtx:
      {.gcsafe.}:
        try: cast[TimerCallback](data.customdata)()
        except:
          when defined(fulldebug): echo "timer: " & getCurrentExceptionMsg()
    else:
      data.socket = fd
      handleRead(gs, data)
      if gs.selector.contains(fd.int):
        try: gs.selector.updateHandle(fd.int, {Event.Read})
        except:
          when defined(fulldebug): echo "updateHandle error: " & getCurrentExceptionMsg()
    currentload.atomicDec()
    if currentload == 0: peakload = 0


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

  var ctxid = 0
  while gs.porthandlers[ctxid.CtxId].port != data.port: ctxid += 1
  gs.selector.registerHandle(fd.int, {Event.Read},
   SocketData(port: gs.porthandlers[ctxid.CtxId].port, ctxid: ctxid.CtxId, socket: fd))

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

  {.gcsafe.}:
    if not gs.multithreading:
      if data.ctxid == TimerCtx:
        {.gcsafe.}:
          try: cast[TimerCallback](data.customdata)()
          except:
            when defined(fulldebug): echo "timer: " & getCurrentExceptionMsg()
      else:
        data.socket = fd
        handleRead(unsafeAddr gs, data)
    else:
      if data.ctxid != TimerCtx:
        try: gs.selector.updateHandle(fd.int, {})
        except:
          echo "removeHandle error: " & getCurrentExceptionMsg()
          continue
      threadgs = unsafeAddr gs
      threadfd = fd
      threaddata = data
      workqueued = true
      signal(workavailable)
      var i = 0
      while workqueued:
        i.inc
        if i mod 8192 == 0: discard sched_yield()
        if i == 131072:
          if shuttingdown: break
          signal(workavailable)
          i = 0
  

proc eventLoop(gs: GuildenServer) {.gcsafe, raises: [].} =
  var eventbuffer: array[1, ReadyKey]
  while true:
    try:
      var ret: int
      try: ret = gs.selector.selectInto(-1, eventbuffer)
      except:
        when defined(fulldebug): echo "select: ", getCurrentExceptionMsg()
        continue

      when defined(fulldebug): echo "event"
      if shuttingdown: break
      if eventbuffer[0].events.len == 0:
        discard sched_yield()
        continue
      
      let event = eventbuffer[0]
      if Event.Signal in event.events:
        echo "Signal event detected..."
        continue
      let fd = posix.SocketHandle(event.fd)
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
        handleEvent()
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
          closeOtherSocket(unsafeAddr gs, data, cause, osErrorMsg(event.errorCode))
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

  {.gcsafe.}:
    if gs.multithreading:
      doAssert(defined(threadsafe), "Selectors module requires compiling with -d:threadsafe")
      if not workerthreadscreated:
        if workerthreadcount == -1: workerthreadcount = countProcessors() + 2
        initCond(workavailable)
        initLock(worklock)
        for i in 1 .. workerthreadcount:
          var thread: Thread[void]
          createThread(thread, threadProc)
        workerthreadscreated = true
        sleep(100)

    var portservers: seq[Socket]
    for i in 0 .. MaxCtxHandlers:
      if gs.porthandlers[i.CtxId].port > 0:
        let port = gs.porthandlers[i.CtxId].port
        let portserver = newSocket()
        portservers.add(portserver)
        try:
          discard setsockopt(posix.SocketHandle(portserver.getFd()), cint(SOL_SOCKET), cint(SO_LINGER), addr linger, SockLen(sizeof(TLinger)))
          portserver.bindAddr(net.Port(port), "")
        except:
          echo "Could not open port ", port
          raise
        gs.selector.registerHandle(portserver.getFd().int, {Event.Read}, SocketData(port: port, ctxid: ServerCtx))
        portserver.listen()
        portserver.getFd().setBlocking(false)

  {.gcsafe.}: signal(SIG_PIPE, SIG_IGN)
  
  sleep(100)
  eventLoop(gs)

  when defined(fulldebug): echo "guildenstern dispatcher loop stopped"
  for portserver in portservers: portserver.close()