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

## .. importdoc:: dispatcher.nim, httpserver.nim, websocketserver.nim, multipartserver.nim

## [GuildenServer] is the abstract base class for upstream (app-facing) web servers. The three concrete server implementations that currently ship
## with GuildenStern are [guildenstern/httpserver], [guildenstern/websocketserver] and [guildenstern/multipartserver].
## One server is associated with one TCP port.
## 
## GuildenServer mainly acts as the glue between everything else, offering set of callback hooks for others to fill in.
## In addition to GuildenServer, this module also introduces SocketContext, which is a container for data of
## one request in flight. SocketContext is inheritable, so concrete servers may add properties to it.
## 
## The overall architecture may be something like this: A reverse proxy (like https://caddyserver.com/) routes requests upsteam to multiple ports.
## Each of these ports is served by one concrete GuildenServer instance. To each server is attached one dispatcher, which listens to the port and
## triggers handlerCallbacks. The default [guildenstern/dispatcher] uses multithreading so that even requests arriving to the same port are served in parallel.
## During request handling, the default servers offer an inherited thread local SocketContext variable from which everything else is accessible,
## most notably the SocketData.server itself and the SocketData.socket that is being serviced.
## 
## Guides for writing your very own servers and dispatchers may appear later. For now, just study the source codes...
## (And if you invent something useful, please share it with us.)
## 

# from std/selectors import newSelectEvent, trigger
from selector import newSelectEvent, trigger
from std/posix import SocketHandle, INVALID_SOCKET, SIGINT, getpid, SIGTERM, onSignal, `==`
from std/net import Socket, newSocket
from std/nativesockets import close
from std/strutils import replace
export SocketHandle, INVALID_SOCKET, posix.`==`

static: doAssert(compileOption("threads"))


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

  LogCallback* = proc(loglevel: LogLevel, message: string) {.gcsafe, nimcall, raises: [].}

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
  shuttingdown* = false ## Global variable that all code is expected to observe and abide to.
  socketcontext* {.threadvar.}: SocketContext
  shutdownevent* = newSelectEvent()
  nextid: int

proc socketdata*(sc: SocketContext): SocketContext {.deprecated:"Use socketcontext directly".} = sc

proc `$`*(x: SocketHandle): string {.inline.} = $(x.cint)

proc shutdown*() =
  ## Sets [shuttingdown] to true and signals dispatcher loops to cease operation.
  {.gcsafe.}:
    shuttingdown = true
    try: trigger(shutdownevent)
    except: discard
 
 
{.hint[XDeclaredButNotUsed]:off.}
onSignal(SIGTERM): shutdown()
onSignal(SIGINT): shutdown()
{.hint[XDeclaredButNotUsed]:on.}


template server(): untyped = socketcontext.server
template thesocket*(): untyped = socketcontext.socket

template log*(theserver: GuildenServer, level: LogLevel, message: string) =
  ## Calls logCallback, if it set. By default, the callback is set to echo the message,
  ## if level is same or higher than server's loglevel.
  if unlikely(int(level) >= int(theserver.loglevel)):
    if likely(not isNil(theserver.logCallback)):
      theserver.logCallback(level, message)


proc initialize*(server: GuildenServer, loglevel: LogLevel) =
  server.id = nextid
  nextid += 1
  server.loglevel = loglevel
  if isNil(server.logCallback): server.logCallback = proc(loglevel: LogLevel, message: string) = (
    block:
      if unlikely(not isNil(getCurrentException())):
        echo LogColors[loglevel.int], loglevel, "\e[0m ", message, ": ", getCurrentExceptionMsg()
      elif message.len < 200: echo LogColors[loglevel.int], loglevel, "\e[0m ", message
      else:
        let excerpt = message[0 .. 49] & " ... (" & $(message.len - 100) & " chars omitted) ... " & message[(message.len - 50) .. (message.len - 1)]
        echo LogColors[loglevel.int], loglevel, "\e[0m ", excerpt.replace("\n", "\\n ")
  )


proc initializeThread*(server: GuildenServer) =
  {.gcsafe.}: server.internalThreadInitializationCallback(server)
  

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


proc suspend*(sleepmillisecs: int) {.inline.} =
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
  logClose(server, socket, cause, msg)
  if not isNil(server.closeSocketCallback):
    server.closeSocketCallback(server, socket, cause, msg)
  else: socket.close()


proc closeOtherSocket*(server: GuildenServer, socket: posix.SocketHandle, cause: SocketCloseCause = CloseCalled, msg: string = "") {.deprecated:"just use closeSocket", gcsafe, nimcall, raises: [].} =
  closeSocket(server, socket, cause, msg)


proc closeSocket*(cause = CloseCalled, msg = "") =
  ## Call this to close the current socket connection.
  closeSocket(server, thesocket, cause, msg)