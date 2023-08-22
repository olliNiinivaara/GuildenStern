## .. importdoc::  dispatcher.nim
## A server for handling HTTP/1.1 traffic.
## 
## The various reply procs and templates should be self-explanatory.
## Exception is the replyStart - replyMore - replyFinish combo.
## They should be used when the response is not available as one chunk (for example,
## due to its size.) In this case, one replyStart starts the response, followed by one ore more replyMores,
## and finally a replyFinish finishes the response.
## 
## Example
## =======
##
## .. code-block:: Nim
##
##  import guildenstern/[dispatcher, httpserver]
##  import httpclient
##     
##  const headerfields = ["afield", "anotherfield"]
##  var headers {.threadvar.}: array[2, string]
##   
##  proc onRequest() =
##    parseHeaders(headerfields, headers)
##    echo headers
##    reply(Http204)
##   
##  let server = newHttpServer(onRequest)
##  server.start(5050)
##  var client = newHttpClient()
##  for i in 1 .. 10: 
##    client.headers = newHttpHeaders({"afield": "value" & $i, "bfield": "bfieldvalue"})
##    discard client.request("http://localhost:5050")
  


from os import sleep, osLastError, osErrorMsg, OSErrorCode
from posix import recv, send, EAGAIN, EWOULDBLOCK, MSG_NOSIGNAL
# from strutils import join
import httpcore
export httpcore 

import guildenserver
export guildenserver


type
  HttpContext* = ref object of SocketContext
    request*: string ## Contains the request itself in [0 ..< requestlen]
    requestlen*: int
    uristart*: int
    urilen*: int
    methlen*: int
    bodystart*: int

  SocketState* = enum
    Fail = -1
    TryAgain = 0
    Progress = 1
    Complete = 2

  HttpServer* = ref object of GuildenServer
    maxheaderlength* = 10000 ## Maximum allowed size for http header part.
    maxrequestlength* = 100000 ## Maximum allowed size for http requests. Every thread will reserve this much memory.
    sockettimeoutms* = 5000 ## If socket is unresponsive for longer, it will be closed.
    requestCallback*: proc(){.gcsafe, nimcall, raises: [].}
    parserequestline*: bool ## If you don't need uri or method, but need max perf, set this to false
    hascontent*: bool ## When serving GET or other method without body, set this to false


when not defined(nimdoc):
  const
    MSG_DONTWAIT* = 0x40.cint
    MSG_MORE* = 0x8000.cint


proc isHttpContext*(): bool = return socketcontext is HttpContext


template http*(): untyped =
  ## Casts the socketcontext thread local variable into a HttpContext
  HttpContext(socketcontext)


template server*(): untyped =
  ## Casts the socketcontext.socketdata.server into a HttpServer
  HttpServer(socketcontext.socketdata.server)


{.push checks: off.}

when not defined(nimdoc):
  proc checkSocketState*(ret: int): SocketState =
    if unlikely(shuttingdown): return Fail
    if likely(ret > 0): return Progress
    if unlikely(ret == 0): return TryAgain
    let lastError = osLastError().int
    let cause =
      if unlikely(ret == Excepted.int): Excepted
      else:
        # https://www-numi.fnal.gov/offline_software/srt_public_context/WebDocs/Errors/unix_system_errors.html
        if lasterror in [EAGAIN.int, EWOULDBLOCK.int]: return TryAgain
        elif lasterror in [2,9]: AlreadyClosed
        elif lasterror == 32: ConnectionLost
        elif lasterror == 104: ClosedbyClient
        else: NetErrored
    if cause == Excepted: closeSocket(Excepted, getCurrentExceptionMsg())
    else: closeSocket(cause, osErrorMsg(OSErrorCode(lastError)))
    return Fail

include httprequest
include httpresponse

when not defined(nimdoc):

  proc prepareHttpContext*(socketdata: ptr SocketData) {.inline.} =
    if unlikely(socketcontext == nil): socketcontext = new HttpContext
    http.socketdata = socketdata
    if unlikely(http.request.len != server.maxrequestlength + 1):
      http.request = newString(server.maxrequestlength + 1)
      if server.threadInitializerCallback != nil: server.threadInitializerCallback(server)
    http.requestlen = 0
    http.uristart = 0
    http.urilen = 0
    http.methlen = 0
    http.bodystart = -1


  proc initHttpServer*(s: HttpServer, loglevel: LogLevel, parserequestline = true, hascontent = true) =
    s.initialize(loglevel)
    s.parserequestline = parserequestline
    s.hascontent = hascontent


  proc handleRequest(data: ptr SocketData) {.gcsafe, nimcall, raises: [].} =
    let socketdata = data[]
    let socketint = socketdata.socket.int
    if unlikely(socketint == -1): return
    prepareHttpContext(addr socketdata)
    if not server.hascontent:
      if not receiveHeader(): return
    else:
      if not receiveAllHttp(): return
    if server.parserequestline and not parseRequestLine(): return
    server.log(DEBUG, "Request of size " & $http.requestlen & " read from socket " & $socketint)
    {.gcsafe.}: server.requestCallback()

{.pop.}


proc newHttpServer*(onrequestcallback: proc(){.gcsafe, nimcall, raises: [].}, loglevel = LogLevel.WARN, parserequestline = true, hascontent = true): HttpServer =
  ## Constructs a new http server. The essential thing here is to set the onrequestcallback proc.
  ## When it is triggered in some thread, that thread offers access to the 
  ## [http] socket context.
  ## 
  ## If you want to tinker with maxheaderlength, maxrequestlength and sockettimeoutms, that is best done
  ## after the server is constructed but before it is started.
  result = new HttpServer
  when not defined(nimdoc):
    result.initHttpServer(loglevel, parserequestline, hascontent)
    result.handlerCallback = handleRequest
    result.requestCallback = onrequestcallback
