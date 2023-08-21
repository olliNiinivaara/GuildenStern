const GuildenSternVersion* = "6.0.0"

#   Guildenstern
#
#  Modular multithreading Linux HTTP + WebSocket server
#
#  (c) Copyright 2020-2023 Olli Niinivaara
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#  
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#


from std/selectors import newSelectEvent, trigger
from std/posix import SocketHandle, INVALID_SOCKET, SIGINT, getpid, SIGTERM, onSignal, `==`
from std/net import Socket, newSocket
from std/nativesockets import close
from std/strutils import replace
export SocketHandle, INVALID_SOCKET, posix.`==`

static: doAssert(compileOption("threads"))


const LogColors = ["\e[90m", "\e[36m", "\e[32m", "\e[34m", "\e[33m", "\e[31m", "\e[35m", "\e[35m"]

type
  GuildenServerException* = Exception

  LogLevel* = enum TRACE, DEBUG, INFO, NOTICE, WARN, ERROR, FATAL, NONE
 
  SocketData* = object
    server*: GuildenServer
    socket*: posix.SocketHandle
    isserversocket*: bool
    customdata*: pointer
    flags*: int

  SocketCloseCause* = enum
    Excepted = -1000
    CloseCalled
    AlreadyClosed
    ClosedbyClient
    ConnectionLost
    TimedOut
    ProtocolViolated
    NetErrored
    SecurityThreatened
    DontClose

  LogCallback* = proc(loglevel: LogLevel, message: string) {.gcsafe, nimcall, raises: [].}
  ThreadInitializerCallback* = proc(server: GuildenServer){.nimcall, gcsafe, raises: [].}
  HandlerCallback* = proc(socketdata: ptr SocketData){.nimcall, gcsafe, raises: [].}
  SuspendCallback* = proc(server: GuildenServer, sleepmillisecs: int){.nimcall, gcsafe, raises: [].}
  CloseSocketCallback* = proc(socketdata: ptr SocketData, cause: SocketCloseCause, msg: string){.gcsafe, nimcall, raises: [].}
  CloseOtherSocketCallback* = proc(server: GuildenServer, socket: posix.SocketHandle, cause: SocketCloseCause, msg: string = ""){.gcsafe, nimcall, raises: [].}
  OnCloseSocketCallback* = proc(socketdata: ptr SocketData, cause: SocketCloseCause, msg: string){.gcsafe, nimcall, raises: [].}


  GuildenServer* {.inheritable.} = ref object
    id*: int
    logCallback*: LogCallback
    loglevel*: LogLevel
    port*: uint16
    thread*: Thread[ptr GuildenServer]
    started*: bool
    threadInitializerCallback*: ThreadInitializerCallback
    handlerCallback*: HandlerCallback
    suspendCallback*: SuspendCallback
    closeSocketCallback*: CloseSocketCallback
    closeOtherSocketCallback*: CloseOtherSocketCallback
    onCloseSocketCallback*: OnCloseSocketCallback


  SocketContext* {.inheritable.} = ref object
    socketdata*: ptr SocketData 


proc `$`*(x: SocketHandle): string {.inline.} = $(x.cint)


var
  shuttingdown* = false
  shutdownevent* = newSelectEvent()
  socketcontext* {.threadvar.}: SocketContext
  nextid: int
  

proc shutdown*() =
  {.gcsafe.}:
    shuttingdown = true
    try: trigger(shutdownevent)
    except: discard
 
 
{.hint[XDeclaredButNotUsed]:off.}
onSignal(SIGTERM): shutdown()
onSignal(SIGINT): shutdown()
{.hint[XDeclaredButNotUsed]:on.}


template log*(server: GuildenServer, level: LogLevel, message: string) =
  if unlikely(int(level) >= int(server.loglevel)):
    if likely(server.logCallback != nil): server.logCallback(level, message)


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


proc closeSocket*(cause = CloseCalled, msg = "") {.gcsafe, nimcall, raises: [].} =
  socketcontext.socketdata.server.closeSocketCallback(socketcontext.socketdata, cause, msg)


proc closeOtherSocket*(server: GuildenServer, socket: posix.SocketHandle, cause: SocketCloseCause = CloseCalled, msg: string = "") {.gcsafe, nimcall, raises: [].} =
  server.closeOtherSocketCallback(server, socket, cause, msg)


proc suspend*(sleepmillisecs: int) {.inline.} =
  socketcontext.socketdata.server.suspendCallback(socketcontext.socketdata.server, sleepmillisecs)


template handleRead*(socketdata: ptr SocketData) =
  {.gcsafe.}: socketdata.server.handlerCallback(socketdata)  