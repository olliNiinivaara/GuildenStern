#
#
#            Nim's Runtime Library
#        (c) Copyright 2016 Eugene Kabanov
#
#    See the file "copying.txt", included in this
#    distribution, for details about the copyright.
#
#    Modified by Olli

## This module allows high-level and efficient I/O multiplexing.
##
## Supported OS primitives: `epoll` and `kqueue`
##
## To use threadsafe version of this module, it needs to be compiled
## with `--threads:on` options.
##
## Supported features: files, sockets, pipes, timers, processes, signals
## and user events.
##
## Supported OS: MacOSX, FreeBSD, OpenBSD, NetBSD, Linux (except
## for Android).
##
## See std/selectors for documentation
##

import std/nativesockets
import std/oserrors

when defined(nimPreviewSlimSystem):
  import std/assertions

const hasThreadSupport = compileOption("threads")

const ioselSupportedPlatform* = defined(macosx) or defined(freebsd) or
                                defined(netbsd) or defined(openbsd) or
                                defined(dragonfly) or defined(nuttx) or
                                (defined(linux) and not defined(android) and not defined(emscripten))

const bsdPlatform {.used.} = defined(macosx) or defined(freebsd) or
                      defined(netbsd) or defined(openbsd) or
                      defined(dragonfly)

import std/strutils

when hasThreadSupport:
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

#[proc setNonBlocking(fd: cint) {.inline.} =
  setBlocking(fd.SocketHandle, false)]#

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

when defined(linux) or defined(macosx) or defined(bsd) or
      defined(solaris) or defined(zephyr) or defined(freertos) or defined(nuttx) or defined(haiku):
  template maxDescriptors*(): int =
    ## Returns the maximum number of active file descriptors for the current
    ## process. This involves a system call. For now `maxDescriptors` is
    ## supported on the following OSes: Windows, Linux, OSX, BSD, Solaris.
    when defined(zephyr) or defined(freertos):
      FD_MAX
    else:
      var fdLim: RLimit
      var res = int(getrlimit(RLIMIT_NOFILE, fdLim))
      if res >= 0:
        res = int(fdLim.rlim_cur) - 1
      res

when defined(nimIoselector):
  when nimIoselector == "epoll":
    include epoll
  elif nimIoselector == "kqueue":
    include ioselects/ioselectors_kqueue
  else:
    {.fatal: "Unknown nimIoselector specified by define.".}
elif defined(linux) and not defined(emscripten):
  include epoll
elif bsdPlatform:
  include ioselects/ioselectors_kqueue
elif defined(nuttx):
  include ioselects/ioselectors_epoll
else:
   {.fatal: "This platform is not supported.".}