const GuildenSternVersion* = "8.0.0"

#   Guildenstern
#
#  Modular multithreading HTTP/1.1 + WebSocket upstream server framework
#
#  (c) Copyright 2020-2025 Olli Niinivaara
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#  
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

# .. importdoc:: httpserver.nim, websocketserver.nim, multipartserver.nim
## [GuildenServer] is the abstract base class for creating server components. The three concrete server implementations that currently ship
## with GuildenStern are guildenstern/httpserver, guildenstern/websocketserver and guildenstern/multipartserver.
## One server is associated with one TCP port, use your internet-facing reverse proxy to route traffic to different servers. 
## 
## GuildenServer itself mainly acts as the glue between everything else, offering set of callback hooks for others to fill in.
##
## In addition to GuildenServer, this module also defines SocketContext, which is a container for data of
## one request in flight, available as the global `socketcontext` threadvar.
## SocketContext is inheritable, so concrete servers may add properties to it.
## 
## To see how to use GuildenStern in practice, consult the various practical examples in the examples folder.
##

from std/posix import SocketHandle, INVALID_SOCKET, SIGINT, getpid, SIGTERM, onSignal, `==`
from std/net import Socket, newSocket
from std/nativesockets import close
from std/strutils import replace
from os import sleep
export SocketHandle, INVALID_SOCKET, posix.`==`


static: doAssert(compileOption("threads"))

func epollSupported*(): bool =
  when defined(linux) and not defined(emscripten): return true
  else:
    when defined(nimIoselector):
      when nimIoselector == "epoll": return true
    
const LogColors = ["\e[90m", "\e[36m", "\e[32m", "\e[34m", "\e[33m", "\e[31m", "\e[35m", "\e[35m"]


type
  LogLevel* = enum TRACE, DEBUG, INFO, NOTICE, WARN, ERROR, FATAL, NONE

  SocketCloseCause* = enum
    ## Parameter in close callbacks.
    EFault = -10000 ## Memory corruption bug
    Excepted = -1000 ## A Nim exception happened
    CloseCalled ## Use this, when the server (your code) closes a socket
    AlreadyClosed  ## Another thread has closed the socket
    ClosedbyClient ## Client closed the connection
    ConnectionLost ## TCP/IP connection was dropped
    TimedOut ## Client did not send/receive all expected data
    ProtocolViolated ## Client was sending garbage
    NetErrored ## Some operating system level error happened
    SecurityThreatened ## Use this, when you decide to close socket for security reasons 
    DontClose ## Internal flag

  LogCallback* = proc(loglevel: LogLevel, source: string, message: string) {.gcsafe, nimcall, raises: [].}

{.warning[Deprecated]:off.}
type
  SocketData* {.deprecated.} = object
    server*: GuildenServer
    socket*: SocketHandle
  
 
  ThreadInitializerCallback* = proc(theserver: GuildenServer){.nimcall, gcsafe, raises: [].}
  ThreadFinalizerCallback* = proc(){.nimcall, gcsafe, raises: [].}
  HandlerCallback* = proc(){.nimcall, gcsafe, raises: [].}
  SuspendCallback* = proc(server: GuildenServer, sleepmillisecs: int){.nimcall, gcsafe, raises: [].}
  CloseSocketCallback* = proc(server: GuildenServer, socket: SocketHandle, cause: SocketCloseCause, msg: string){.gcsafe, nimcall, raises: [].}
  OnCloseSocketCallback* = proc(server: GuildenServer, socket: SocketHandle, cause: SocketCloseCause, msg: string){.gcsafe, nimcall, raises: [].} ## The `msg` parameter may contain furher info about the cause. For example, in case of websocket ClosedByClient, `msg` contains the status code.]#
  DeprecatedOnCloseSocketCallback* {.deprecated:"use OnCloseSocketCallback".} = proc(socketdata: ptr SocketData, cause: SocketCloseCause, msg: string){.gcsafe, nimcall, raises: [].}
  GetFlagsCallback* = proc(server: GuildenServer, socket: SocketHandle): int {.nimcall, gcsafe, raises: [].}
  SetFlagsCallback* = proc(server: GuildenServer, socket: SocketHandle, newflags: int): bool {.nimcall, gcsafe, raises: [].}


  GuildenServer* {.inheritable.} = ref object
    port*: uint16
    thread*: Thread[GuildenServer]
    id*: int
    logCallback*: LogCallback
    loglevel*: LogLevel
    started*: bool
    internalThreadInitializationCallback*: ThreadInitializerCallback
    threadInitializerCallback*: ThreadInitializerCallback
    threadFinalizerCallback*: ThreadFinalizerCallback
    handlerCallback*: HandlerCallback
    suspendCallback*: SuspendCallback
    closeSocketCallback*: CloseSocketCallback
    onCloseSocketCallback*: OnCloseSocketCallback
    deprecatedOnCloseSocketCallback*: DeprecatedOnCloseSocketCallback
    getFlagsCallback*: GetFlagsCallback
    setFlagsCallback*: SetFlagsCallback 

  SocketContext* {.inheritable.} = ref object
    server*: GuildenServer
    socket*: SocketHandle
    customdata*: pointer
{.warning[Deprecated]:on.}

var
  shuttingdown* = false ## Global variable that all code is expected to observe and abide to (check this inside your loops every now and then...).
  socketcontext* {.threadvar.}: SocketContext
  ## Access to data relating to the current socket request. Various servers offer templates to access their own more specialized data, such as the `http` and `ws` templates of the httpserver and websocketserver, respectively.
  nextid: int
  shutdownCallbacks*: seq[proc() {.nimcall, gcsafe, raises: [].}]


proc socketdata*(sc: SocketContext): SocketContext {.deprecated:"Use socketcontext directly".} = sc

proc `$`*(x: SocketHandle): string {.inline.} = $(x.cint)

proc shutdown*() =
  ## Sets [shuttingdown] to true and signals dispatcher loops to cease operation.
  {.gcsafe.}:
    shuttingdown = true
    for shutdownCallback in shutdownCallbacks: shutdownCallback()
 
 
{.hint[XDeclaredButNotUsed]:off.}
onSignal(SIGTERM): shutdown()
onSignal(SIGINT): shutdown()
{.hint[XDeclaredButNotUsed]:on.}

template thesocket*(): untyped =
  ## Global shortcut for accessing `socketcontext.socket`
  socketcontext.socket

template log*(theserver: GuildenServer, level: LogLevel, message: string) =
  ## Calls logCallback, if it set. By default, the callback is set to echo the message,
  ## if level is same or higher than server's loglevel.
  if unlikely(int(level) >= int(theserver.loglevel)):
    if likely(not isNil(theserver.logCallback)):
      var s = if theserver.port == 0: "c" else: "s"
      s.add($theserver.id & " " & $getThreadId())
      theserver.logCallback(level, s, message)


template log*(theserver: GuildenServer, level: LogLevel, source: string, message: string) =
  ## Calls logCallback, if it set. By default, the callback is set to echo the message,
  ## if level is same or higher than server's loglevel.
  if unlikely(int(level) >= int(theserver.loglevel)):
    if likely(not isNil(theserver.logCallback)):
      theserver.logCallback(level, source, message)


proc initialize*(server: GuildenServer, loglevel: LogLevel) =
  server.id = nextid
  nextid += 1
  server.loglevel = loglevel
  if isNil(server.logCallback): server.logCallback = proc(loglevel: LogLevel, source: string, message: string) {.nimcall.} = (
    block:
      if unlikely(not isNil(getCurrentException())):
        echo LogColors[loglevel.int], loglevel, "\e[0m ", source, " ", message, ": ", getCurrentExceptionMsg()
      elif message.len < 200: echo LogColors[loglevel.int], loglevel, "\e[0m ", source, " ", message
      else:
        let excerpt = message[0 .. 49] & " ... (" & $(message.len - 100) & " chars omitted) ... " & message[(message.len - 50) .. (message.len - 1)]
        echo LogColors[loglevel.int], loglevel, "\e[0m ", source, " ", excerpt.replace("\n", "\\n ")
  )


proc initializeThread*(server: GuildenServer) =
  {.gcsafe.}:
    if not isNil(server.internalThreadInitializationCallback):
      server.internalThreadInitializationCallback(server)
  

proc handleRead*(theserver: GuildenServer, socket: SocketHandle, customdata: pointer) =
  {.gcsafe.}:
    if unlikely(socket == INVALID_SOCKET): return
    socketcontext.server = theserver
    socketcontext.socket = socket
    socketcontext.customdata = customdata
    theserver.handlerCallback()


proc getFlags*(server: GuildenServer, socket: posix.SocketHandle): int =
  assert not isNil(server.getFlagsCallback)
  return server.getFlagsCallback(server, socket)

proc setFlags*(server: GuildenServer, socket: posix.SocketHandle, flags: int): bool =
  assert not isNil(server.setFlagsCallback)
  return server.setFlagsCallback(server, socket, flags)


proc suspend*(sleepmillisecs: int) {.deprecated:"use suspend that takes server as parameter".} =
  sleep(sleepmillisecs)


proc suspend*(server: GuildenServer, sleepmillisecs: int) {.inline.} =
  # If operation gets stalled, use this instead of sleep, so that the dispatcher may also react (and the suspend get logged)
  if not isNil(server.suspendCallback):
    server.suspendCallback(server, sleepmillisecs)


proc logClose(server: GuildenServer, socket: SocketHandle, cause = CloseCalled, msg: string) =
  let loglevel =
    if cause in [CloseCalled, AlreadyClosed, ClosedbyClient]: DEBUG
    elif cause in [ConnectionLost, TimedOut]: INFO
    elif cause in [ProtocolViolated, NetErrored]: NOTICE
    elif cause  == SecurityThreatened: WARN
    else: ERROR
  server.log(loglevel, "socket " & $socket & " " & $cause & ": " & msg) 


proc closeSocket*(server: GuildenServer, socket: SocketHandle, cause = CloseCalled, msg = "") {.gcsafe, nimcall, raises: [].} =
  ## Call this to close any socket connection.
  if socket == INVALID_SOCKET: return
  logClose(server, socket, cause, msg)
  if not isNil(server.closeSocketCallback):
    server.closeSocketCallback(server, socket, cause, msg)
  else: discard posix.close(socket)


proc closeOtherSocket*(server: GuildenServer, socket: posix.SocketHandle, cause: SocketCloseCause = CloseCalled, msg: string = "") {.deprecated:"just use closeSocket", gcsafe, nimcall, raises: [].} =
  closeSocket(server, socket, cause, msg)


proc closeSocket*(cause = CloseCalled, msg = "") =
  ## Call this to close the current socket connection.
  if not isNil(socketcontext.server):
    closeSocket(socketcontext.server, thesocket, cause, msg)