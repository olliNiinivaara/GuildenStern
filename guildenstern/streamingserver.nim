## Server for cases when request body must be processed as it's being received (for example, when client is uploading a large dataset). 
## TODO: multipart download (server -> client), parallel processing of parts 

import std/[times, monotimes]
from os import osLastError, osErrorMsg, OSErrorCode, sleep
from posix import recv, SocketHandle
from strutils import find, startsWith
import httpserver
export httpserver

type
  FileState = enum Before, In, After

  StreamingHandler* = ref object of HttpHandler
    contentlength*: int64
    contentreceived*: int64
    contentdelivered*: int64
    waitms: int
    timeout: Duration
    start: MonoTime
    boundary: string
    filestate: FileState
  
  
proc isStreamingHandler*(): bool = return guildenhandler is StreamingHandler

template stream*(): untyped = StreamingHandler(guildenhandler)


proc receiveStreamingHeader(): bool {.gcsafe, raises:[].} =
  while true:
    if shuttingdown: return false
    let ret = recv(http.socketdata.socket, addr http.request[http.requestlen], 1 + server.maxheaderlength - http.requestlen, 0)
    if checkSocketState(ret) == Fail: return false
    http.requestlen += ret
    if isHeaderreceived(http.requestlen - ret, http.requestlen): break
    if http.requestlen > server.maxheaderlength:
      closeSocket(ProtocolViolated, "stream receiveHeader: Max header size exceeded")
      return false
  stream.contentlength = getContentLength()
  stream.contentreceived = http.requestlen - http.bodystart
  true


proc hasData*(): bool  =
  return stream.contentlength > 0 and stream.contentdelivered < stream.contentlength


proc receiveChunk*(): SocketState {.gcsafe, raises:[] .} =
  if shuttingdown: return Fail
  if stream.contentdelivered == 0 and stream.contentreceived > 0:
    stream.contentdelivered = stream.requestlen - stream.bodystart
    stream.requestlen = stream.contentdelivered.int
    return Progress
  let ret = recv(stream.socketdata.socket, addr stream.request[0], (stream.contentlength - stream.contentreceived).int, MSG_DONTWAIT)
  result = checkSocketState(ret)
  stream.contentreceived += ret
  stream.contentdelivered += ret
  stream.requestlen = ret


proc startReceiveMultipart*(waitMs = 100, giveupSecs = 10): bool {.gcsafe, raises: [].} =
  const contenttype = ["content-type"]
  var value: array[1, string]
  parseHeaders(contenttype, value)
  if not value[0].startsWith("multipart/form-data; boundary="): return false
  stream.waitms = waitMs
  stream.timeout = initDuration(seconds = giveupSecs)
  stream.start = getMonoTime()
  stream.filestate = Before
  stream.boundary = "--" & value[0][30 .. value[0].high] & "--"
  return true


proc receiveMultipart*(): (SocketState , string) {.gcsafe, raises: [].} =
  if stream.filestate == After or not hasData(): return (Complete , "")
  let state = receiveChunk()
  if state == Fail: return (Fail , "")
  if state == TryAgain:
    sleep(stream.waitms)
    if getMonoTime() - stream.start > stream.timeout: return (Fail, "")
    return (TryAgain , "")

  var filestart = 0
  if stream.filestate == Before:
    filestart = stream.request.find("\c\L\c\L") + 4
    if filestart == 3: return (TryAgain , "")
    else: stream.filestate = In
  
  var fileend = stream.request.find(stream.boundary, filestart) - 2
  if fileend == -3:
    fileend = stream.requestlen
    return (Progress , stream.request[filestart .. fileend - filestart])
  else:
    while true:
      if receiveChunk() != Progress: break
    stream.filestate = After
    return (Progress , stream.request[filestart .. fileend - filestart])


proc handleStreamRequest(data: ptr SocketData) {.gcsafe, nimcall, raises: [].} =
  if (unlikely)guildenhandler == nil: guildenhandler = new StreamingHandler
  prepareHttpHandler(data)
  stream.contentlength = -1
  stream.contentreceived = 0
  stream.contentdelivered = 0
  if receiveStreamingHeader():
    if parseRequestLine():
      {.gcsafe.}: HttpServer(data.server).requestCallback()
    

proc newStreamingServer*(onrequestcallback: proc(){.gcsafe, nimcall, raises: [].}, loglevel = LogLevel.WARN): HttpServer =
  result = new HttpServer
  result.initHttpServer(loglevel, true, true, true)
  result.handlerCallback = handleStreamRequest
  result.requestCallback = onrequestcallback