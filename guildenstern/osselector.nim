#
#
#            Nim's Runtime Library
#        (c) Copyright 2016 Eugene Kabanov
#
#    See the file "copying.txt", included in this
#    distribution, for details about the copyright.
#
#    Modified by Olli

import std/nativesockets
import std/oserrors

when defined(nimPreviewSlimSystem):
  import std/assertions

const ioselSupportedPlatform* = defined(macosx) or defined(freebsd) or
                                defined(netbsd) or defined(openbsd) or
                                defined(dragonfly) or defined(nuttx) or
                                (defined(linux) and not defined(android) and not defined(emscripten))
  ## This constant is used to determine whether the destination platform is
  ## fully supported by `ioselectors` module.

#[const bsdPlatform = defined(macosx) or defined(freebsd) or
                    defined(netbsd) or defined(openbsd) or
                    defined(dragonfly)]#

when defined(nimdoc):
  discard
else:
  import std/strutils

  import std/locks

  type
    SharedArray[T] = UncheckedArray[T]

  proc allocSharedArray[T](nsize: int): ptr SharedArray[T] =
    result = cast[ptr SharedArray[T]](allocShared0(sizeof(T) * nsize))

  proc reallocSharedArray[T](sa: ptr SharedArray[T], oldsize, nsize: int): ptr SharedArray[T] =
    result = cast[ptr SharedArray[T]](reallocShared0(sa, oldsize * sizeof(T), sizeof(T) * nsize))

  proc deallocSharedArray[T](sa: ptr SharedArray[T]) =
    deallocShared(cast[pointer](sa))
    
  type
    Event* {.pure.} = enum
      Read, Write, Timer, Signal, Process, Vnode, User, Error, Oneshot,
      Finished, VnodeWrite, VnodeDelete, VnodeExtend, VnodeAttrib, VnodeLink,
      VnodeRename, VnodeRevoke

  type
    IOSelectorsException* = object of CatchableError

    ReadyKey* = object
      fd*: int
      events*: set[Event]
      errorCode*: OSErrorCode

    SelectorKey[T] = object
      ident: int
      events: set[Event]
      param: int
      data: T

  const
    InvalidIdent = -1

  proc raiseIOSelectorsError[T](message: T) =
    var msg = ""
    when T is string:
      msg.add(message)
    elif T is OSErrorCode:
      msg.add(osErrorMsg(message) & " (code: " & $int(message) & ")")
    else:
      msg.add("Internal Error\n")
    var err = newException(IOSelectorsException, msg)
    raise err

  #proc setNonBlocking(fd: cint) {.inline.} =
  #  setBlocking(fd.SocketHandle, false)

  when not defined(windows):
    import std/posix

    template setKey(s, pident, pevents, pparam, pdata: untyped) =
      var skey = addr(s.fds[pident])
      skey.ident = pident
      skey.events = pevents
      skey.param = pparam
      skey.data = pdata

  when ioselSupportedPlatform:
    template blockSignals(newmask: var Sigset, oldmask: var Sigset) =
      when hasThreadSupport:
        if posix.pthread_sigmask(SIG_BLOCK, newmask, oldmask) == -1:
          raiseIOSelectorsError(osLastError())
      else:
        if posix.sigprocmask(SIG_BLOCK, newmask, oldmask) == -1:
          raiseIOSelectorsError(osLastError())

    template unblockSignals(newmask: var Sigset, oldmask: var Sigset) =
      when hasThreadSupport:
        if posix.pthread_sigmask(SIG_UNBLOCK, newmask, oldmask) == -1:
          raiseIOSelectorsError(osLastError())
      else:
        if posix.sigprocmask(SIG_UNBLOCK, newmask, oldmask) == -1:
          raiseIOSelectorsError(osLastError())

  template clearKey[T](key: ptr SelectorKey[T]) =
    var empty: T
    key.ident = InvalidIdent
    key.events = {}
    key.data = empty

  proc verifySelectParams(timeout: int) =
    # Timeout of -1 means: wait forever
    # Anything higher is the time to wait in milliseconds.
    doAssert(timeout >= -1, "Cannot select with a negative value, got: " & $timeout)

  when defined(linux) or defined(windows) or defined(macosx) or defined(bsd) or
       defined(solaris) or defined(zephyr) or defined(freertos) or defined(nuttx) or defined(haiku):
    template maxDescriptors*(): int =
      ## Returns the maximum number of active file descriptors for the current
      ## process. This involves a system call. For now `maxDescriptors` is
      ## supported on the following OSes: Windows, Linux, OSX, BSD, Solaris.
      when defined(windows):
        16_700_000
      elif defined(zephyr) or defined(freertos):
        FD_MAX
      else:
        var fdLim: RLimit
        var res = int(getrlimit(RLIMIT_NOFILE, fdLim))
        if res >= 0:
          res = int(fdLim.rlim_cur) - 1
        res


#
#
#            Nim's Runtime Library
#        (c) Copyright 2016 Eugene Kabanov
#
#    See the file "copying.txt", included in this
#    distribution, for details about the copyright.
#
#    Modified by Olli

# This module implements Linux epoll().

import std/[times, epoll]

# Maximum number of events that can be returned
const MAX_EPOLL_EVENTS = 64

when not defined(android):
  type
    SignalFdInfo* {.importc: "struct signalfd_siginfo",
                    header: "<sys/signalfd.h>", pure, final.} = object
      ssi_signo*: uint32
      ssi_errno*: int32
      ssi_code*: int32
      ssi_pid*: uint32
      ssi_uid*: uint32
      ssi_fd*: int32
      ssi_tid*: uint32
      ssi_band*: uint32
      ssi_overrun*: uint32
      ssi_trapno*: uint32
      ssi_status*: int32
      ssi_int*: int32
      ssi_ptr*: uint64
      ssi_utime*: uint64
      ssi_stime*: uint64
      ssi_addr*: uint64
      pad* {.importc: "__pad".}: array[0..47, uint8]

proc timerfd_create(clock_id: ClockId, flags: cint): cint
     {.cdecl, importc: "timerfd_create", header: "<sys/timerfd.h>".}
proc timerfd_settime(ufd: cint, flags: cint,
                      utmr: var Itimerspec, otmr: var Itimerspec): cint
     {.cdecl, importc: "timerfd_settime", header: "<sys/timerfd.h>".}
proc eventfd(count: cuint, flags: cint): cint
     {.cdecl, importc: "eventfd", header: "<sys/eventfd.h>".}

when not defined(android):
  proc signalfd(fd: cint, mask: var Sigset, flags: cint): cint
       {.cdecl, importc: "signalfd", header: "<sys/signalfd.h>".}

type
  SelectorImpl[T] = object
    epollFD: cint
    maxFD: int
    numFD: int
    fds: ptr SharedArray[SelectorKey[T]]
    count*: int
  Selector*[T] = ptr SelectorImpl[T]

type
  SelectEventImpl = object
    efd: cint
  SelectEvent* = ptr SelectEventImpl

proc newSelector*[T](): Selector[T] =
  proc initialNumFD(): int {.inline.} =
    when defined(nuttx):
      result = NEPOLL_MAX
    else:
      result = 1024
  # Retrieve the maximum fd count (for current OS) via getrlimit()
  var maxFD = maxDescriptors()
  doAssert(maxFD > 0)
  # Start with a reasonable size, checkFd() will grow this on demand
  let numFD = initialNumFD()

  var epollFD = epoll_create1(O_CLOEXEC)
  if epollFD < 0:
    raiseOSError(osLastError())

  result = cast[Selector[T]](allocShared0(sizeof(SelectorImpl[T])))
  result.epollFD = epollFD
  result.maxFD = maxFD
  result.numFD = numFD
  result.fds = allocSharedArray[SelectorKey[T]](numFD)

  for i in 0 ..< numFD:
    result.fds[i].ident = InvalidIdent

proc close*[T](s: Selector[T]) =
  let res = posix.close(s.epollFD)
  deallocSharedArray(s.fds)
  deallocShared(cast[pointer](s))
  if res != 0:
    raiseIOSelectorsError(osLastError())

proc newSelectEvent*(): SelectEvent =
  let fdci = eventfd(0, O_CLOEXEC or O_NONBLOCK)
  if fdci == -1:
    raiseIOSelectorsError(osLastError())
  result = cast[SelectEvent](allocShared0(sizeof(SelectEventImpl)))
  result.efd = fdci

proc trigger*(ev: SelectEvent) =
  var data: uint64 = 1
  if posix.write(ev.efd, addr data, sizeof(uint64)) == -1:
    raiseIOSelectorsError(osLastError())

proc close*(ev: SelectEvent) =
  let res = posix.close(ev.efd)
  deallocShared(cast[pointer](ev))
  if res != 0:
    raiseIOSelectorsError(osLastError())

#[template checkFd(s, f) =
  # TODO: I don't see how this can ever happen. You won't be able to create an
  # FD if there is too many. -- DP
  if f >= s.maxFD:
    raiseIOSelectorsError("Maximum number of descriptors is exhausted!")
  if f >= s.numFD:
    var numFD = s.numFD
    while numFD <= f: numFD *= 2
    s.fds = reallocSharedArray(s.fds, s.numFD, numFD)
    for i in s.numFD ..< numFD:
      s.fds[i].ident = InvalidIdent
    s.numFD = numFD]#

template checkFd(s, f) =
  if f >= s.numFD:
    var numFD = s.numFD
    while numFD <= f: numFD *= 2
    s.fds = reallocSharedArray(s.fds, s.numFD, numFD)
    for i in s.numFD ..< numFD: s.fds[i].ident = InvalidIdent
    s.numFD = numFD


proc registerHandle*[T](s: Selector[T], fd: int | SocketHandle,
                        events: set[Event], data: T) =
  let fdi = int(fd)
  s.checkFd(fdi)
  doAssert(s.fds[fdi].ident == InvalidIdent, "Descriptor $# already registered" % $fdi)
  s.setKey(fdi, events, 0, data)
  if events != {}:
    var epv = EpollEvent(events: EPOLLRDHUP)
    epv.data.u64 = fdi.uint
    if Event.Read in events: epv.events = epv.events or EPOLLIN
    if Event.Write in events: epv.events = epv.events or EPOLLOUT
    if epoll_ctl(s.epollFD, EPOLL_CTL_ADD, fdi.cint, addr epv) != 0:
      raiseIOSelectorsError(osLastError())
    inc(s.count)


proc registerExclusiveReadHandle*[T](s: Selector[T], fd: int | SocketHandle,  data: T) =
  let fdi = int(fd)
  s.checkFd(fdi)
  doAssert(s.fds[fdi].ident == InvalidIdent, "Descriptor $# already registered" % $fdi)
  s.setKey(fdi, {Event.Read}, 0, data)
  var epv = EpollEvent(events: EPOLLRDHUP or EPOLLIN or EPOLLET)
  epv.data.u64 = fdi.uint
  if epoll_ctl(s.epollFD, EPOLL_CTL_ADD, fdi.cint, addr epv) != 0:
    raiseIOSelectorsError(osLastError())
  inc(s.count)
  
  
proc updateHandle*[T](s: Selector[T], fd: int | SocketHandle, events: set[Event]) =
  let maskEvents = {Event.Timer, Event.Signal, Event.Process, Event.Vnode,
                    Event.User, Event.Oneshot, Event.Error}
  let fdi = int(fd)
  s.checkFd(fdi)
  var pkey = addr(s.fds[fdi])
  doAssert(pkey.ident != InvalidIdent,
           "Descriptor $# is not registered in the selector!" % $fdi)
  doAssert(pkey.events * maskEvents == {})
  if pkey.events != events:
    var epv = EpollEvent(events: EPOLLRDHUP)
    epv.data.u64 = fdi.uint

    if Event.Read in events: epv.events = epv.events or EPOLLIN
    if Event.Write in events: epv.events = epv.events or EPOLLOUT

    if pkey.events == {}:
      if epoll_ctl(s.epollFD, EPOLL_CTL_ADD, fdi.cint, addr epv) != 0:
        raiseIOSelectorsError(osLastError())
      inc(s.count)
    else:
      if events != {}:
        if epoll_ctl(s.epollFD, EPOLL_CTL_MOD, fdi.cint, addr epv) != 0:
          raiseIOSelectorsError(osLastError())
      else:
        if epoll_ctl(s.epollFD, EPOLL_CTL_DEL, fdi.cint, addr epv) != 0:
          raiseIOSelectorsError(osLastError())
        dec(s.count)
    pkey.events = events

proc unregister*[T](s: Selector[T], fd: int|SocketHandle) =
  let fdi = int(fd)
  s.checkFd(fdi)
  var pkey = addr(s.fds[fdi])
  doAssert(pkey.ident != InvalidIdent,
           "Descriptor $# is not registered in the selector!" % $fdi)
  if pkey.events != {}:
    when not defined(android):
      if Event.Read in pkey.events or Event.Write in pkey.events or Event.User in pkey.events:
        var epv = EpollEvent()
        # TODO: Refactor all these EPOLL_CTL_DEL + dec(s.count) into a proc.
        if epoll_ctl(s.epollFD, EPOLL_CTL_DEL, fdi.cint, addr epv) != 0:
          raiseIOSelectorsError(osLastError())
        dec(s.count)
      elif Event.Timer in pkey.events:
        if Event.Finished notin pkey.events:
          var epv = EpollEvent()
          if epoll_ctl(s.epollFD, EPOLL_CTL_DEL, fdi.cint, addr epv) != 0:
            raiseIOSelectorsError(osLastError())
          dec(s.count)
        if posix.close(cint(fdi)) != 0:
          raiseIOSelectorsError(osLastError())
      elif Event.Signal in pkey.events:
        var epv = EpollEvent()
        if epoll_ctl(s.epollFD, EPOLL_CTL_DEL, fdi.cint, addr epv) != 0:
          raiseIOSelectorsError(osLastError())
        var nmask, omask: Sigset
        discard sigemptyset(nmask)
        discard sigemptyset(omask)
        discard sigaddset(nmask, cint(s.fds[fdi].param))
        unblockSignals(nmask, omask)
        dec(s.count)
        if posix.close(cint(fdi)) != 0:
          raiseIOSelectorsError(osLastError())
      elif Event.Process in pkey.events:
        if Event.Finished notin pkey.events:
          var epv = EpollEvent()
          if epoll_ctl(s.epollFD, EPOLL_CTL_DEL, fdi.cint, addr epv) != 0:
            raiseIOSelectorsError(osLastError())
          var nmask, omask: Sigset
          discard sigemptyset(nmask)
          discard sigemptyset(omask)
          discard sigaddset(nmask, SIGCHLD)
          unblockSignals(nmask, omask)
          dec(s.count)
        if posix.close(cint(fdi)) != 0:
          raiseIOSelectorsError(osLastError())
    else:
      if Event.Read in pkey.events or Event.Write in pkey.events or Event.User in pkey.events:
        var epv = EpollEvent()
        if epoll_ctl(s.epollFD, EPOLL_CTL_DEL, fdi.cint, addr epv) != 0:
          raiseIOSelectorsError(osLastError())
        dec(s.count)
      elif Event.Timer in pkey.events:
        if Event.Finished notin pkey.events:
          var epv = EpollEvent()
          if epoll_ctl(s.epollFD, EPOLL_CTL_DEL, fdi.cint, addr epv) != 0:
            raiseIOSelectorsError(osLastError())
          dec(s.count)
        if posix.close(cint(fdi)) != 0:
          raiseIOSelectorsError(osLastError())
  clearKey(pkey)

proc unregister*[T](s: Selector[T], ev: SelectEvent) =
  let fdi = int(ev.efd)
  s.checkFd(fdi)
  var pkey = addr(s.fds[fdi])
  doAssert(pkey.ident != InvalidIdent, "Event is not registered in the queue!")
  doAssert(Event.User in pkey.events)
  var epv = EpollEvent()
  if epoll_ctl(s.epollFD, EPOLL_CTL_DEL, fdi.cint, addr epv) != 0:
    raiseIOSelectorsError(osLastError())
  dec(s.count)
  clearKey(pkey)

proc registerTimer*[T](s: Selector[T], timeout: int, oneshot: bool,
                       data: T): int {.discardable.} =
  var
    newTs: Itimerspec
    oldTs: Itimerspec
  let fdi = timerfd_create(CLOCK_MONOTONIC, O_CLOEXEC or O_NONBLOCK).int
  if fdi == -1:
    raiseIOSelectorsError(osLastError())

  s.checkFd(fdi)
  doAssert(s.fds[fdi].ident == InvalidIdent)

  var events = {Event.Timer}
  var epv = EpollEvent(events: EPOLLIN or EPOLLRDHUP)
  epv.data.u64 = fdi.uint

  if oneshot:
    newTs.it_interval.tv_sec = posix.Time(0)
    newTs.it_interval.tv_nsec = 0
    newTs.it_value.tv_sec = posix.Time(timeout div 1_000)
    newTs.it_value.tv_nsec = (timeout %% 1_000) * 1_000_000
    incl(events, Event.Oneshot)
    epv.events = epv.events or EPOLLONESHOT
  else:
    newTs.it_interval.tv_sec = posix.Time(timeout div 1000)
    newTs.it_interval.tv_nsec = (timeout %% 1_000) * 1_000_000
    newTs.it_value.tv_sec = newTs.it_interval.tv_sec
    newTs.it_value.tv_nsec = newTs.it_interval.tv_nsec

  if timerfd_settime(fdi.cint, cint(0), newTs, oldTs) != 0:
    raiseIOSelectorsError(osLastError())
  if epoll_ctl(s.epollFD, EPOLL_CTL_ADD, fdi.cint, addr epv) != 0:
    raiseIOSelectorsError(osLastError())
  s.setKey(fdi, events, 0, data)
  inc(s.count)
  result = fdi

when not defined(android):
  proc registerSignal*[T](s: Selector[T], signal: int,
                          data: T): int {.discardable.} =
    var
      nmask: Sigset
      omask: Sigset

    discard sigemptyset(nmask)
    discard sigemptyset(omask)
    discard sigaddset(nmask, cint(signal))
    blockSignals(nmask, omask)

    let fdi = signalfd(-1, nmask, O_CLOEXEC or O_NONBLOCK).int
    if fdi == -1:
      raiseIOSelectorsError(osLastError())

    s.checkFd(fdi)
    doAssert(s.fds[fdi].ident == InvalidIdent)

    var epv = EpollEvent(events: EPOLLIN or EPOLLRDHUP)
    epv.data.u64 = fdi.uint
    if epoll_ctl(s.epollFD, EPOLL_CTL_ADD, fdi.cint, addr epv) != 0:
      raiseIOSelectorsError(osLastError())
    s.setKey(fdi, {Event.Signal}, signal, data)
    inc(s.count)
    result = fdi

  proc registerProcess*[T](s: Selector, pid: int,
                           data: T): int {.discardable.} =
    var
      nmask: Sigset
      omask: Sigset

    discard sigemptyset(nmask)
    discard sigemptyset(omask)
    discard sigaddset(nmask, posix.SIGCHLD)
    blockSignals(nmask, omask)

    let fdi = signalfd(-1, nmask, O_CLOEXEC or O_NONBLOCK).int
    if fdi == -1:
      raiseIOSelectorsError(osLastError())

    s.checkFd(fdi)
    doAssert(s.fds[fdi].ident == InvalidIdent)

    var epv = EpollEvent(events: EPOLLIN or EPOLLRDHUP)
    epv.data.u64 = fdi.uint
    epv.events = EPOLLIN or EPOLLRDHUP
    if epoll_ctl(s.epollFD, EPOLL_CTL_ADD, fdi.cint, addr epv) != 0:
      raiseIOSelectorsError(osLastError())
    s.setKey(fdi, {Event.Process, Event.Oneshot}, pid, data)
    inc(s.count)
    result = fdi

proc registerEvent*[T](s: Selector[T], ev: SelectEvent, data: T) =
  let fdi = int(ev.efd)
  doAssert(s.fds[fdi].ident == InvalidIdent, "Event is already registered in the queue!")
  s.setKey(fdi, {Event.User}, 0, data)
  var epv = EpollEvent(events: EPOLLIN or EPOLLRDHUP)
  epv.data.u64 = ev.efd.uint
  if epoll_ctl(s.epollFD, EPOLL_CTL_ADD, ev.efd, addr epv) != 0:
    raiseIOSelectorsError(osLastError())
  inc(s.count)


proc selectInto*[T](s: Selector[T], timeout: int,
                    results: var openArray[ReadyKey]): int =
  var
    resTable: array[MAX_EPOLL_EVENTS, EpollEvent]
    maxres = MAX_EPOLL_EVENTS
    i, k: int

  if maxres > len(results):
    maxres = len(results)

  verifySelectParams(timeout)

  let count = epoll_wait(s.epollFD, addr(resTable[0]), maxres.cint,
                         timeout.cint)
  if count < 0:
    result = 0
    let err = osLastError()
    if cint(err) != EINTR:
      raiseIOSelectorsError(err)
  elif count == 0:
    result = 0
  else:
    i = 0
    k = 0
    while i < count:
      let fdi = int(resTable[i].data.u64)
      let pevents = resTable[i].events
      var pkey = addr(s.fds[fdi])
      doAssert(pkey.ident != InvalidIdent)
      var rkey = ReadyKey(fd: fdi, events: {})

      if (pevents and EPOLLERR) != 0 or (pevents and EPOLLHUP) != 0:
        if (pevents and EPOLLHUP) != 0:
          rkey.errorCode = OSErrorCode ECONNRESET
        else:
          # Try reading SO_ERROR from fd.
          var error: cint
          var size = SockLen sizeof(error)
          if getsockopt(SocketHandle fdi, SOL_SOCKET, SO_ERROR, addr(error),
                        addr(size)) == 0'i32:
            rkey.errorCode = OSErrorCode error

        rkey.events.incl(Event.Error)
      if (pevents and EPOLLOUT) != 0:
        rkey.events.incl(Event.Write)
      when not defined(android):
        if (pevents and EPOLLIN) != 0:
          if Event.Read in pkey.events:
            rkey.events.incl(Event.Read)
          elif Event.Timer in pkey.events:
            var data: uint64 = 0
            if posix.read(cint(fdi), addr data,
                          sizeof(uint64)) != sizeof(uint64):
              raiseIOSelectorsError(osLastError())
            rkey.events.incl(Event.Timer)
          elif Event.Signal in pkey.events:
            var data = SignalFdInfo()
            if posix.read(cint(fdi), addr data,
                          sizeof(SignalFdInfo)) != sizeof(SignalFdInfo):
              raiseIOSelectorsError(osLastError())
            rkey.events.incl(Event.Signal)
          elif Event.Process in pkey.events:
            var data = SignalFdInfo()
            if posix.read(cint(fdi), addr data,
                          sizeof(SignalFdInfo)) != sizeof(SignalFdInfo):
              raiseIOSelectorsError(osLastError())
            if cast[int](data.ssi_pid) == pkey.param:
              rkey.events.incl(Event.Process)
            else:
              inc(i)
              continue
          elif Event.User in pkey.events:
            var data: uint64 = 0
            if posix.read(cint(fdi), addr data,
                          sizeof(uint64)) != sizeof(uint64):
              let err = osLastError()
              if err == OSErrorCode(EAGAIN):
                inc(i)
                continue
              else:
                raiseIOSelectorsError(err)
            rkey.events.incl(Event.User)
      else:
        if (pevents and EPOLLIN) != 0:
          if Event.Read in pkey.events:
            rkey.events.incl(Event.Read)
          elif Event.Timer in pkey.events:
            var data: uint64 = 0
            if posix.read(cint(fdi), addr data,
                          sizeof(uint64)) != sizeof(uint64):
              raiseIOSelectorsError(osLastError())
            rkey.events.incl(Event.Timer)
          elif Event.User in pkey.events:
            var data: uint64 = 0
            if posix.read(cint(fdi), addr data,
                          sizeof(uint64)) != sizeof(uint64):
              let err = osLastError()
              if err == OSErrorCode(EAGAIN):
                inc(i)
                continue
              else:
                raiseIOSelectorsError(err)
            rkey.events.incl(Event.User)

      if Event.Oneshot in pkey.events:
        var epv = EpollEvent()
        if epoll_ctl(s.epollFD, EPOLL_CTL_DEL, cint(fdi), addr epv) != 0:
          raiseIOSelectorsError(osLastError())
        # we will not clear key until it will be unregistered, so
        # application can obtain data, but we will decrease counter,
        # because epoll is empty.
        dec(s.count)
        # we are marking key with `Finished` event, to avoid double decrease.
        pkey.events.incl(Event.Finished)

      results[k] = rkey
      inc(k)
      inc(i)
    result = k



proc selectFast*[T](s: Selector[T]): ReadyKey =
  var res: EpollEvent
  while true:
    let count = epoll_wait(s.epollFD, addr res, 1.cint, -1.cint)
    if count != 1: continue
    let fdi = int(res.data.u64)
    let pkey = addr(s.fds[fdi])
    if pkey.ident == InvalidIdent:
      echo "invalid ident"
      continue

    result = ReadyKey(fd: fdi, events: {})

    if (res.events and EPOLLERR) != 0 or (res.events and EPOLLHUP) != 0 or (res.events and EPOLLRDHUP) != 0: # or (res.events and EPOLLRDHUP)
      if (res.events and EPOLLHUP) != 0:
        result.errorCode = OSErrorCode ECONNRESET
      else:
        var error: cint
        var size = SockLen sizeof(error)
        if getsockopt(SocketHandle fdi, SOL_SOCKET, SO_ERROR, addr(error), addr(size)) == 0'i32:
          result.errorCode = OSErrorCode error
      result.events.incl(Event.Error)
    if (res.events and EPOLLIN) != 0: result.events.incl(Event.Read)
    elif Event.User in pkey.events: result.events.incl(Event.User)
    if result.events.len > 0: break



proc select*[T](s: Selector[T], timeout: int): seq[ReadyKey] =
  result = newSeq[ReadyKey](MAX_EPOLL_EVENTS)
  let count = selectInto(s, timeout, result)
  result.setLen(count)

template isEmpty*[T](s: Selector[T]): bool =
  (s.count == 0)

proc contains*[T](s: Selector[T], fd: SocketHandle|int): bool {.inline.} =
  return s.fds[fd.int].ident != InvalidIdent

proc getData*[T](s: Selector[T], fd: SocketHandle|int): var T =
  let fdi = int(fd)
  s.checkFd(fdi)
  if fdi in s:
    result = s.fds[fdi].data


proc getSafelyData*[T](default: var T, s: Selector[T], fd: SocketHandle|int): var T =
  let fdi = int(fd)
  s.checkFd(fdi)
  if fdi in s:
    result = s.fds[fdi].data
  else:
    result = default


proc setData*[T](s: Selector[T], fd: SocketHandle|int, data: T): bool =
  let fdi = int(fd)
  s.checkFd(fdi)
  if fdi in s:
    s.fds[fdi].data = data
    result = true

template withData*[T](s: Selector[T], fd: SocketHandle|int, value,
                        body: untyped) =
  mixin checkFd
  let fdi = int(fd)
  s.checkFd(fdi)
  if fdi in s:
    var value = addr(s.fds[fdi].data)
    body

template withData*[T](s: Selector[T], fd: SocketHandle|int, value, body1,
                        body2: untyped) =
  mixin checkFd
  let fdi = int(fd)
  s.checkFd(fdi)
  if fdi in s:
    var value = addr(s.fds[fdi].data)
    body1
  else:
    body2

proc getFd*[T](s: Selector[T]): int =
  return s.epollFD.int
