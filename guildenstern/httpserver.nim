## HTTP server, with three operating modes
## 
## see examples/httptest.nim, examples/streamingposttest.nim, and examples/replychunkedtest.nim for examples.
##

from os import sleep, osLastError, osErrorMsg, OSErrorCode
from posix import recv, send, EAGAIN, EWOULDBLOCK, MSG_NOSIGNAL
import httpcore
export httpcore
import strtabs
export strtabs

import guildenserver
export guildenserver

type
  ContentType* = enum
    ## mode of the server
    NoBody ## offers slightly faster handling for requests like GET that do not have a body
    Compact ## the default mode. Whole request body must fit into the request string (size defined with [bufferlength] parameter), from where it can then be accessed with [getRequest], [isBody] and [getBody] procs
    Streaming ## read the body yourself with the [receiveStream] iterator 

  HttpContext* = ref object of SocketContext
    request*: string
    requestlen*: int64
    uristart*: int64
    urilen*: int64
    methlen*: int64
    bodystart*: int64
    contentlength*: int64
    contentreceived*: int64
    contentdelivered*: int64
    headers*: StringTableRef
    probebuffer: string

  SocketState* = enum
    Fail = -1
    TryAgain = 0
    Progress = 1
    Complete = 2

  HttpServerObj* = object of GuildenServer
    contenttype*: ContentType
    maxheaderlength*: int64 ## Maximum allowed size for http header part.
    bufferlength*: int64 ## Every thread will reserve this much memory, for buffering the incoming request. Must be larger than maxheaderlength.
    sockettimeoutms*: int ## If socket is unresponsive for longer, it will be closed.
    requestCallback*: proc(){.gcsafe, nimcall, raises: [].}
    parserequestline*: bool ## If you don't need uri or method, but need max perf, set this to false
    headerfields*: seq[string] ## list of header fields to be parsed

  HttpServer* = ptr HttpServerObj

const
  MSG_DONTWAIT* = when defined(macosx): 0x80.cint else: 0x40.cint
  MSG_MORE* = 0x8000.cint
  initialbackoff* = 1


proc isHttpContext*(): bool = return socketcontext is HttpContext

template http*(): untyped =
  ## Casts the socketcontext thread local variable into a HttpContext
  HttpContext(socketcontext)

template server*(): untyped =
  cast[HttpServer](socketcontext.server)

when defined(release):
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
      elif lasterror == 14: EFault
      elif lasterror == 32: ConnectionLost
      elif lasterror == 104: ClosedbyClient
      else: NetErrored
  if cause == Excepted: closeSocket(Excepted)
  else: closeSocket(cause, osErrorMsg(OSErrorCode(lastError)))
  return Fail


include httprequest
include httpresponse


proc handleHttpThreadInitialization*(theserver: GuildenServer) =
  if socketcontext.isNil: socketcontext = new HttpContext
  let thisserver = cast[HttpServer](theserver)
  http.request = newString(thisserver.bufferlength + 1)
  http.probebuffer = newString(1)
  if thisserver.contenttype != NoBody or thisserver.headerfields.len > 0:
    http.headers = newStringTable()
    for field in thisserver.headerfields: http.headers[field] = ""
    if not http.headers.contains("content-length"): http.headers["content-length"] = ""
  if not isNil(theserver.threadInitializerCallback): theserver.threadInitializerCallback(theserver)
  

proc prepareHttpContext*() {.inline.} =
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
  s.name = "HTTP-" & $s.id
  s.maxheaderlength = 10000.int64
  s.bufferlength = 100000.int64
  s.sockettimeoutms = 5000
  s.contenttype = contenttype
  s.parserequestline = parserequestline
  s.headerfields = newSeq[string](headerfields.len)
  s.headerfields.add(headerfields)
  if isNil(s.internalThreadInitializationCallback): s.internalThreadInitializationCallback = handleHttpThreadInitialization


proc handleRequest() {.gcsafe, nimcall, raises: [].} =
  prepareHttpContext()
  if not readHeader(): return
  if server.parserequestline and not parseRequestLine(): return
  case server.contenttype:
    of NoBody:
        server.log(DEBUG, "Nobody request of length " & $http.requestlen & " read from socket " & $thesocket)
    of Compact:
      if http.contentlength > server.bufferlength:
          closeSocket(ProtocolViolated, "content-length larger than bufferlength")
          return
      if not receiveToSingleBuffer():
        server.log(DEBUG, "Receiving request to single buffer failed from socket " & $thesocket)
        return
    of Streaming:
      server.log(DEBUG, "Started request streaming with chunk of length " & $http.requestlen & " from socket " & $thesocket)
  {.gcsafe.}:
    if likely(not isNil(server.requestCallback)): server.requestCallback()

when defined(release):
  {.pop.}


proc newHttpServer*(onrequestcallback: proc(){.gcsafe, nimcall, raises: [].}, loglevel = LogLevel.WARN, parserequestline = true, contenttype = Compact, headerfields: openArray[string] = []): HttpServer =
  ## Constructs a new http server. The essential thing here is to set the onrequestcallback proc.
  ## When it is triggered, the [http] thread-local socket context is accessible.
  ## 
  ## If you want to tinker with [HttpServer.maxheaderlength], [HttpServer.bufferlength] or [HttpServer.sockettimeoutms], that is best done
  ## after the server is constructed but before it is started.
  result = cast[HttpServer](allocShared0(sizeof(HttpServerObj)))
  result.initHttpServer(loglevel, parserequestline, contenttype, headerfields)
  for field in result[].headerfields:
    for c in field:
      if c != '-' and not isLowerAscii(c):
        result.log(ERROR, "Header field not in lower case: " & field)
  result.handlerCallback = handleRequest
  result.requestCallback = onrequestcallback