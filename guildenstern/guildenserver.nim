const GuildenSternVersion* = "7.3.0"

#   Guildenstern
#
#  Modular multithreading HTTP/1.1 + WebSocket upstream server framework
#
#  (c) Copyright 2020-2024 Olli Niinivaara
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

from std/selectors import newSelectEvent, trigger
from std/posix import SocketHandle, INVALID_SOCKET, SIGINT, getpid, SIGTERM, onSignal, `==`
from std/net import Socket, newSocket
when not defined(nimdoc): from std/nativesockets import close
from std/strutils import replace
export SocketHandle, INVALID_SOCKET, posix.`==`

static: doAssert(compileOption("threads"))


const LogColors = ["\e[90m", "\e[36m", "\e[32m", "\e[34m", "\e[33m", "\e[31m", "\e[35m", "\e[35m"]

type
  LogLevel* = enum TRACE, DEBUG, INFO, NOTICE, WARN, ERROR, FATAL, NONE

  SocketCloseCause* = enum
    ## Parameter in close callbacks.
    Excepted = -1000 ## A Nim exception happened
    CloseCalled ## Use this, when the server (your code) closes a socket
    AlreadyClosed  ## Another thread has closed the socket
    ClosedbyClient ## Client closed the connection
    ConnectionLost ## TCP/IP connection was dropped
    TimedOut ## Client did not send/receive all expected data
    ProtocolViolated ## Client was sending garbage
    NetErrored ## Some operating system level error happened
    SecurityThreatened ## Use this, when you decide to close socket for security reasosns 
    DontClose ## Internal flag

  LogCallback* = proc(loglevel: LogLevel, message: string) {.gcsafe, nimcall, raises: [].}

type
  SocketData* = object
    ## | Data associated with every incoming socket message.
    ## | This is available in [SocketContext] via ``socketdata`` pointer.
    ## | customdata pointer can be freely used in user code.
    server*: GuildenServer
    socket*: SocketHandle
    isserversocket*: bool
    flags*: int
    customdata*: pointer

  ThreadInitializerCallback* = proc(server: GuildenServer){.nimcall, gcsafe, raises: [].}
  HandlerCallback* = proc(socketdata: ptr SocketData){.nimcall, gcsafe, raises: [].}
  SuspendCallback* = proc(server: GuildenServer, sleepmillisecs: int){.nimcall, gcsafe, raises: [].}
  CloseSocketCallback* = proc(socketdata: ptr SocketData, cause: SocketCloseCause, msg: string){.gcsafe, nimcall, raises: [].}
  CloseOtherSocketCallback* = proc(server: GuildenServer, socket: SocketHandle, cause: SocketCloseCause, msg: string = ""){.gcsafe, nimcall, raises: [].}
  OnCloseSocketCallback* = proc(socketdata: ptr SocketData, cause: SocketCloseCause, msg: string){.gcsafe, nimcall, raises: [].} ## The `msg` parameter may contain furher info about the cause. For example, in case of websocket ClosedByClient, `msg` contains the status code. 


  GuildenServer* {.inheritable.} = ref object
    port*: uint16
    thread*: Thread[ptr GuildenServer]
    id*: int
    logCallback*: LogCallback
    loglevel*: LogLevel
    started*: bool
    internalThreadInitializationCallback*: ThreadInitializerCallback
    threadInitializerCallback*: ThreadInitializerCallback
    handlerCallback*: HandlerCallback
    suspendCallback*: SuspendCallback
    closeSocketCallback*: CloseSocketCallback
    closeOtherSocketCallback*: CloseOtherSocketCallback
    onCloseSocketCallback*: OnCloseSocketCallback

  SocketContext* {.inheritable.} = ref object
    socketdata*: ptr SocketData  

var
  shuttingdown* = false ## Global variable that all code is expected to observe and abide to.
  socketcontext* {.threadvar.}: SocketContext
  shutdownevent* = newSelectEvent()
  nextid: int


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


template log*(theserver: GuildenServer, level: LogLevel, message: string) =
  ## Calls logCallback, if it set. By default, the callback is set to echo the message,
  ## if level is same or higher than server's loglevel.
  if unlikely(int(level) >= int(theserver.loglevel)):
    if likely(theserver.logCallback != nil):
      theserver.logCallback(level, message)


proc initialize*(server: GuildenServer, loglevel: LogLevel) =
  server.id = nextid
  nextid += 1
  server.loglevel = loglevel
  if server.logCallback == nil: server.logCallback = proc(loglevel: LogLevel, message: string) = (
    block:
      if unlikely(getCurrentException() != nil):
        echo LogColors[loglevel.int], loglevel, "\e[0m ", message, ": ", getCurrentExceptionMsg()
      elif message.len < 200: echo LogColors[loglevel.int], loglevel, "\e[0m ", message
      else:
        let excerpt = message[0 .. 49] & " ... (" & $(message.len - 100) & " chars omitted) ... " & message[(message.len - 50) .. (message.len - 1)]
        echo LogColors[loglevel.int], loglevel, "\e[0m ", excerpt.replace("\n", "\\n ")
  )


template initializeThread*(server: ptr GuildenServer) =
  {.gcsafe.}: server.internalThreadInitializationCallback(server[])
  

template handleRead*(socketdata: ptr SocketData) =
  {.gcsafe.}: socketdata.server.handlerCallback(socketdata) 


proc closeSocket*(cause = CloseCalled, msg = "") {.gcsafe, nimcall, raises: [].} =
  ## Call this to close a socket yourself.
  socketcontext.socketdata.server.closeSocketCallback(socketcontext.socketdata, cause, msg)


proc closeOtherSocket*(server: GuildenServer, socket: posix.SocketHandle, cause: SocketCloseCause = CloseCalled, msg: string = "") {.gcsafe, nimcall, raises: [].} =
  ## Call this to close an open socket that is not the socket currently being served.
  server.closeOtherSocketCallback(server, socket, cause, msg)


proc suspend*(sleepmillisecs: int) {.inline.} =
  ## Instead of os.sleep, use this. This informs a dispatcher that your thread is waiting for something and therefore
  ## another thread should be allowed to run.
  socketcontext.socketdata.server.suspendCallback(socketcontext.socketdata.server, sleepmillisecs) 