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
    sockettimeoutms*: int
    requestCallback*: proc(){.gcsafe, nimcall, raises: [].}
    parserequestline*: bool
    parseheaders*: bool
    hascontent*: bool

  HttpHandler* = ref object of GuildenHandler
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


const
  MSG_DONTWAIT* = 0x40.cint
  MSG_MORE* = 0x8000.cint


proc isHttpHandler*(): bool = return guildenhandler is HttpHandler

template http*(): untyped = HttpHandler(guildenhandler)

template server*(): untyped = HttpServer(guildenhandler.socketdata.server)

{.push checks: off.}

proc prepareHttpHandler*(socketdata: ptr SocketData) {.inline.} =
  if unlikely(guildenhandler == nil): guildenhandler = new HttpHandler
  http.socketdata = socketdata
  if unlikely(http.request.len != server.maxrequestlength + 1):
    http.request = newString(server.maxrequestlength + 1)
    if server.threadInitializerCallback != nil: server.threadInitializerCallback(server)
  if server.parseheaders and http.headers == nil: http.headers = newStringTable()
  http.requestlen = 0
  http.uristart = 0
  http.urilen = 0
  http.methlen = 0
  http.bodystart = -1


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


proc handleRequest(data: ptr SocketData) {.gcsafe, nimcall, raises: [].} =
  let socketdata = data[]
  let socketint = socketdata.socket.int
  if unlikely(socketint == -1): return
  prepareHttpHandler(addr socketdata)
  if not server.hascontent:
    if not receiveHeader(): return
  else:
    if not receiveAllHttp(): return
  if server.parserequestline and not parseRequestLine(): return
  if server.parseHeaders: parseHeaders(http.headers)
  server.log(DEBUG, "Request of size " & $http.requestlen & " read from socket " & $socketint)
  {.gcsafe.}: server.requestCallback()

{.pop.}


proc initHttpServer*(s: HttpServer, loglevel: LogLevel, parserequestline = true, parseheaders = true, hascontent = true) =
  s.initialize(loglevel)
  s.maxheaderlength = 10000
  s.maxrequestlength = 100000
  s.sockettimeoutms = 5000
  s.parserequestline = parserequestline
  s.parseheaders = parseheaders
  s.hascontent = hascontent


proc newHttpServer*(onrequestcallback: proc(){.gcsafe, nimcall, raises: [].}, loglevel = LogLevel.WARN, parserequestline = true, parseheaders = true, hascontent = true): HttpServer =
  result = new HttpServer
  result.initHttpServer(loglevel, parserequestline, parseheaders, hascontent)
  result.handlerCallback = handleRequest
  result.requestCallback = onrequestcallback
