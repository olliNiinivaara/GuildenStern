from os import sleep, osLastError, osErrorMsg, OSErrorCode
from posix import recv, send, EAGAIN, EWOULDBLOCK, MSG_NOSIGNAL
import httpcore
export httpcore
import strtabs
export strtabs

import guildenserver
export guildenserver

type
  ContentType* = enum NoBody, Compact, Streaming

  HttpContext* = ref object of SocketContext
    request*: string
    requestlen*: int
    uristart*: int
    urilen*: int
    methlen*: int
    bodystart*: int
    contentlength*: int64
    contentreceived*: int64
    contentdelivered*: int64
    waitms: int
    headers*: StringTableRef

  SocketState* = enum
    Fail = -1
    TryAgain = 0
    Progress = 1
    Complete = 2

  HttpServer* = ref object of GuildenServer
    contenttype*: ContentType
    maxheaderlength* = 10000 ## Maximum allowed size for http header part.
    bufferlength* = 100000 ## Every thread will reserve this much memory, for buffering the incoming request. Must be larger than maxheaderlength.
    sockettimeoutms* = 5000 ## If socket is unresponsive for longer, it will be closed.
    requestCallback*: proc(){.gcsafe, nimcall, raises: [].}
    parserequestline*: bool ## If you don't need uri or method, but need max perf, set this to false
    headerfields*: seq[string] # = @["content-length"] ## list of header fields to be parsed. content-length must always be kept included. 

const
  MSG_DONTWAIT* = when defined(macosx): 0x80.cint else: 0x40.cint
  MSG_MORE* = 0x8000.cint


proc isHttpContext*(): bool = return socketcontext is HttpContext


template http*(): untyped =
  ## Casts the socketcontext thread local variable into a HttpContext
  HttpContext(socketcontext)


template server*(): untyped =
  ## Casts the socketcontext.socketdata.server into a HttpServer
  HttpServer(socketcontext.socketdata.server)


{.push checks: off.}

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


proc prepareHttpContext*(socketdata: ptr SocketData) {.inline.} =
  if unlikely(socketcontext == nil): socketcontext = new HttpContext
  http.socketdata = socketdata
  if unlikely(http.request.len != server.bufferlength + 1):
    http.request = newString(server.bufferlength + 1)
    if server.headerfields.len > 0:
      http.headers = newStringTable()
      for field in server.headerfields: http.headers[field] = ""
      if server.contenttype != NoBody and not http.headers.contains("content-length"): http.headers["content-length"] = ""
    if server.threadInitializerCallback != nil: server.threadInitializerCallback(server)
  http.requestlen = 0
  http.contentlength = 0
  http.uristart = 0
  http.urilen = 0
  http.methlen = 0
  http.bodystart = -1
  if server.headerfields.len > 0:
    try:
      for key in http.headers.keys: http.headers[key].setLen(0)
    except:
      echo "header key error, should never happen"


proc initHttpServer*(s: HttpServer, loglevel: LogLevel, parserequestline: bool, contenttype: ContentType, headerfields: openArray[string]) =
  s.initialize(loglevel)
  s.contenttype = contenttype
  s.parserequestline = parserequestline
  s.headerfields.add(headerfields)


proc handleRequest(data: ptr SocketData) {.gcsafe, nimcall, raises: [].} =
  let socketdata = data[]
  let socketint = socketdata.socket.int
  if unlikely(socketint == -1): return
  prepareHttpContext(addr socketdata)
  if not readHeader(): return
  if server.parserequestline and not parseRequestLine(): return
  case server.contenttype:
    of NoBody:
        server.log(DEBUG, "Nobody request of length " & $http.requestlen & " read from socket " & $socketint)
    of Compact:
      if http.contentlength > server.bufferlength:
          closeSocket(ProtocolViolated, "content-length larger than bufferlength")
          return
      if not receiveToSingleBuffer():
        server.log(DEBUG, "Receiving request to single buffer failed from socket " & $socketint)
        return
    of Streaming:
      server.log(DEBUG, "Started request streaming with chunk of length " & $http.requestlen & " from socket " & $socketint)
  {.gcsafe.}: server.requestCallback()

{.pop.}


proc newHttpServer*(onrequestcallback: proc(){.gcsafe, nimcall, raises: [].}, loglevel = LogLevel.WARN, parserequestline = true, contenttype = Compact, headerfields: openArray[string] = []): HttpServer =
  ## Constructs a new http server. The essential thing here is to set the onrequestcallback proc.
  ## When it is triggered in some thread, that thread offers access to the 
  ## [http] socket context.
  ## 
  ## If you want to tinker with maxheaderlength, maxrequestlength and sockettimeoutms, that is best done
  ## after the server is constructed but before it is started.
  for field in headerfields:
    for c in field:
      if not isLowerAscii(c):
        echo "Header field not in lower case: ", field
        quit()
  result = new HttpServer
  result.initHttpServer(loglevel, parserequestline, contenttype, headerfields)
  result.handlerCallback = handleRequest
  result.requestCallback = onrequestcallback