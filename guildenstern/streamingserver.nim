## Server for cases when request body must be processed as it's being received (for example, when client is uploading a large dataset).
##
## Example
## =======
## 
## .. code-block:: Nim
## 
##  import guildenstern/[dispatcher, streamingserver]
##  
##  let html = """<!doctype html><title>File Upload</title><body>
##    <form action="/upload" method="post" enctype="multipart/form-data" accept-charset="utf-8">
##    <input type="file" id="file" name="file">
##    <input type="submit">"""; let ok = "ok"
##  
##  proc handleGet() =
##    while hasData(): echo receiveChunk()
##    reply(Http200, html)
##  
##  proc handleUpload() =
##    if not startReceiveMultipart(): (reply(Http400); return)
##    while true:
##      let (state , chunk) = receiveMultipart()
##      case state
##        of Fail: break
##        of TryAgain: continue
##        of Progress: echo chunk
##        of Complete: (reply(Http200, ok); break)
##    shutdown()
##  
##  proc onRequest() =
##    {.gcsafe.}:
##      if startsUri("/upload"): handleUpload()
##      else: handleGet()
##      
##  var server = newStreamingServer(onRequest)
##  server.start(5050)
##  joinThread(server.thread)
##

import std/[times, monotimes]
from posix import recv, SocketHandle
from strutils import find, startsWith
import httpserver
export httpserver

type
  FileState = enum Before, In, After

  StreamingContext* = ref object of HttpContext
    contentlength*: int64
    contentreceived*: int64
    contentdelivered*: int64
    waitms: int
    timeout: Duration
    start: MonoTime
    boundary: string
    filestate: FileState
  
  
proc isStreamingContext*(): bool = return socketcontext is StreamingContext

template stream*(): untyped =
  ## Casts the socketcontext thread local variable into a StreamingContext
  StreamingContext(socketcontext)


when not defined(nimdoc):
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
  ## While this is true, there are more chunks to receive
  return stream.contentlength > 0 and stream.contentdelivered < stream.contentlength


proc receiveChunk*(): SocketState {.gcsafe, raises:[] .} =
  ## Tries to receive a chunk, and returns the state of the socket stream.
  when defined(nimdoc): discard
  else:
    if shuttingdown: return Fail
    if stream.contentdelivered == 0 and stream.contentreceived > 0:
      stream.contentdelivered = stream.requestlen - stream.bodystart
      stream.requestlen = stream.contentdelivered.int
      return Progress
    let ret = recv(stream.socketdata.socket, addr stream.request[0], (stream.contentlength - stream.contentreceived).int, MSG_DONTWAIT)
    result = checkSocketState(ret)
    if result == Fail: return Fail
    stream.contentreceived += ret
    stream.contentdelivered += ret
    stream.requestlen = ret
    if not hasData(): result = Complete 


proc startReceiveMultipart*(waitMs = 100, giveupSecs = 10): bool {.gcsafe, raises: [].} =
  ## Starts receiving a multipart/form-data http request as chuncks.
  ## This is how browsers deliver file uploads to server, see example.
  ## 
  ## `waitMs` is time to sleep before [receiveMultipart] returns `TryAgain`.
  ## `giveupSecs` is total time for receiving before socket is closed.
  ## 
  ##  Returns false if content-type header is not multipart/form-data.
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
  ## Tries to receive a chunk from a stream started with [startReceiveMultipart].
  ## Returns both the socket's state and the possibly received chunk.
  if stream.filestate == After or not hasData(): return (Complete , "")
  if getMonoTime() - stream.start > stream.timeout:
      closeSocket(TimedOut, "reached giveupSecs = " & $stream.timeout)
      return (Fail, "")
  let state = receiveChunk()
  if state == Fail: return (Fail , "")
  if state == TryAgain:
    suspend(stream.waitms)
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

when not defined(nimdoc):
  proc handleStreamRequest(data: ptr SocketData) {.gcsafe, nimcall, raises: [].} =
    if (unlikely)socketcontext == nil: socketcontext = new StreamingContext
    prepareHttpContext(data)
    stream.contentlength = -1
    stream.contentreceived = 0
    stream.contentdelivered = 0
    if receiveStreamingHeader():
      if parseRequestLine():
        {.gcsafe.}: HttpServer(data.server).requestCallback()
    

proc newStreamingServer*(onrequestcallback: proc(){.gcsafe, nimcall, raises: [].}, loglevel = LogLevel.WARN): HttpServer =
  result = new HttpServer
  when not defined(nimdoc):
    result.initHttpServer(loglevel, true, true)
    result.handlerCallback = handleStreamRequest
  result.requestCallback = onrequestcallback