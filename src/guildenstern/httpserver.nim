from os import sleep, osLastError, osErrorMsg, OSErrorCode
from posix import recv, send, EAGAIN, EWOULDBLOCK, MSG_NOSIGNAL
from strutils import find, parseInt, isLowerAscii, toLowerAscii, join
import httpcore
export httpcore
import strtabs
export strtabs
import guildenserver
export guildenserver


type
  HttpServer* = ref object of GuildenServer
    maxheaderlength*: int
    maxrequestlength*: int
    blockingsendtimeoutms*: int
    requestCallback*: proc(){.gcsafe, nimcall, raises: [].}
    parserequestline*: bool
    parseheaders*: bool
    hascontent*: bool

  HttpHandler* {.inheritable.} = ref object of GuildenHandler
    request*: string
    requestlen*: int
    uristart*: int
    urilen*: int
    methlen*: int
    bodystart*: int
    headers*: StringTableRef

  SocketState* = enum
    Fail = -1
    TryAgain = 0
    Progress = 1
    Complete = 2


proc isHttpHandler*(): bool = return guildenhandler is HttpHandler

template http*(): untyped = HttpHandler(guildenhandler)

template server*(): untyped = HttpServer(guildenhandler.socketdata.server)


proc prepareHttpHandler*(socketdata: ptr SocketData) {.inline.} =
  if unlikely(guildenhandler == nil): guildenhandler = new HttpHandler
  http.socketdata = socketdata
  if unlikely(http.request.len != server.maxrequestlength + 1): http.request = newString(server.maxrequestlength + 1)
  if server.parseheaders and http.headers == nil: http.headers = newStringTable()
  http.requestlen = 0
  http.uristart = 0
  http.urilen = 0
  http.methlen = 0
  http.bodystart = -1


{.push checks: off.}

proc checkSocketState*(ret: int, nonblocking = false): SocketState =
  if unlikely(shuttingdown): return Fail
  if likely(ret > 0): return Progress
  if unlikely(ret == 0): return TryAgain
  let lastError = osLastError().int
  let cause =
    if unlikely(ret == Excepted.int): Excepted
    else:
      # https://www-numi.fnal.gov/offline_software/srt_public_context/WebDocs/Errors/unix_system_errors.html
      if lasterror in [EAGAIN.int, EWOULDBLOCK.int]:
        if nonblocking: return TryAgain
        else: TimedOut
      elif lasterror in [2,9]: AlreadyClosed
      elif lasterror == 32: ConnectionLost
      elif lasterror == 104: ClosedbyClient
      else: NetErrored
  if cause == Excepted: server.closeSocket(http.socketdata, Excepted, getCurrentExceptionMsg())
  else:  server.closeSocket(http.socketdata, cause, osErrorMsg(OSErrorCode(lastError)))
  return Fail


include httprequest
include httpresponse


proc handleRequest(data: ptr SocketData) {.gcsafe, nimcall, raises: [].} =
  prepareHttpHandler(data)
  if not server.hascontent:
    if not receiveHeader(): return
  else:
    if not receiveAllHttp(): return
  if server.parserequestline and not parseRequestLine(): return
  if server.parseHeaders: parseHeaders(http.headers)

  {.gcsafe.}: server.requestCallback()


{.pop.}


proc initHttpServer*(s: HttpServer, parserequestline = true, parseheaders = true, hascontent = true) =
  s.maxheaderlength = 10000
  s.maxrequestlength = 100000
  s.blockingsendtimeoutms = 3000
  s.parserequestline = parserequestline
  s.parseheaders = parseheaders
  s.hascontent = hascontent


proc newHttpServer*(onrequestcallback: proc(){.gcsafe, nimcall, raises: [].}, parserequestline = true, parseheaders = true, hascontent = true): HttpServer =
  result = new HttpServer
  result.initHttpServer(parserequestline, parseheaders, hascontent)
  result.registerHandler(handleRequest)
  result.requestCallback = onrequestcallback
