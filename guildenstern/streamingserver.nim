## Server for cases when request body must be processed as a stream (for example, when client is uploading a large dataset to server), or
## response must sent as a stream (for example, when client is downloading a large dataset from server.)
##
## Example
## =======
## 
## .. code-block:: Nim
##  
##  import guildenstern/[dispatcher, streamingserver]
##  
##  proc handleUpload() =
##    let ok = "ok"
##    if not startReceiveMultipart(giveupSecs = 2): (reply(Http400); return)
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
##    let html = """<!doctype html><title>StreamCtx</title><body>
##    <form action="/upload" method="post" enctype="multipart/form-data" accept-charset="utf-8">
##    <input type="file" id="file" name="file">
##    <input type="submit">"""
##    
##    if startsUri("/upload"): handleUpload()
##    else: reply(Http200, html)
##      
##  let server = newStreamingServer(onRequest)
##  server.start(5050)
##  joinThread(server.thread)
##

import std/[times, monotimes]
from posix import recv, SocketHandle
from strutils import find, startsWith
from strformat import fmt
import httpserver
export httpserver

type
  FileState = enum Before, In, After

when defined(nimdoc):
  type StreamingContext* = ref object of HttpContext
else:
  type 
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
    var backoff = 1
    var totalbackoff = 0
    while true:
      if shuttingdown: return false
      let ret = recv(http.socketdata.socket, addr http.request[http.requestlen], 1 + server.maxheaderlength - http.requestlen, MSG_DONTWAIT)
      let state = checkSocketState(ret)
      if state == Fail: return false
      if state == SocketState.TryAgain:
        suspend(backoff)
        totalbackoff += backoff
        backoff *= 2
        if totalbackoff > server.sockettimeoutms:
          closeSocket(TimedOut, "didn't receive streaming header")
          return false
        continue
      http.requestlen += ret
      if isHeaderreceived(http.requestlen - ret, http.requestlen): break
      if http.requestlen > server.maxheaderlength:
        closeSocket(ProtocolViolated, "stream receiveHeader: Max header size exceeded")
        return false
    stream.contentlength = getContentLength()
    stream.contentreceived = http.requestlen - http.bodystart
    true


proc hasData*(): bool  =
  ## While this is true, there are more multipart chunks to receive
  when defined(nimdoc): discard
  else: return stream.contentlength > 0 and stream.contentdelivered < stream.contentlength


proc receiveChunk(): SocketState {.gcsafe, raises:[] .} =
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
  ## Starts receiving a multipart/form-data http request as chunks.
  ## This is how browsers deliver file uploads to server, see example above.
  ## 
  ## `waitMs` is time to sleep before [receiveMultipart] returns `TryAgain`.
  ## `giveupSecs` is total time for receiving before socket is closed.
  ## 
  ##  Returns false if content-type header is not multipart/form-data.
  ##
  when defined(nimdoc): discard
  else:
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
  when defined(nimdoc): discard
  else:
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


iterator receiveInChunks*(): (SocketState , string) {.gcsafe, raises: [].} =
  ## Receives a http request in chunks, yielding the state of operation and a possibly received new chuck on every iteration.
  ## With this, you can receive POST data without worries about main memory usage.
  ## See examples/streamingposttest.nim for a concrete working example of how to use this iterator.
  when defined(nimdoc): discard
  else:
    stream.contentlength = getContentLength()
    stream.contentreceived = stream.requestlen - stream.bodystart
    if stream.contentlength == 0: yield (Complete , "")
    else:
      if stream.contentreceived == stream.contentlength:
        yield (Progress , stream.request[stream.bodystart ..< stream.bodystart + stream.contentlength])
        yield (Complete , "")
      else:
        yield (Progress , stream.request[stream.bodystart ..< stream.bodystart + stream.contentreceived])
        var continues = true
        while continues:
          if shuttingdown:
            yield (Fail , "")
            continues = false
          else:
            let recvsize =
              if stream.contentlength - stream.contentreceived > server.maxrequestlength: server.maxrequestlength
              else: stream.contentlength - stream.contentreceived
            let ret = recv(stream.socketdata.socket, addr stream.request[0], recvsize, MSG_DONTWAIT)
            let state = checkSocketState(ret)
            stream.requestlen = ret
            stream.contentreceived += ret
            if state == Fail:
              yield (Fail , "")
              continues = false
            elif state == TryAgain:
              yield (TryAgain , "")
            elif state == Complete or stream.contentlength == stream.contentreceived:
              yield(Progress , stream.request[0 ..< ret])
              yield(Complete , "")
              continues = false
            else: yield(Progress , stream.request[0 ..< ret])


proc startDownload*(code: HttpCode = Http200, headers: openArray[string] = [], waitMs = 100, giveupSecs = 10): bool =
  ## Starts replying http response as `Transfer-encoding: chunked`.
  ## Allows sending large datasets in multiple parts so that main memory is not exhausted.
  ## Also supports sending dynamic data, where Content-length header cannot be set.
  ## 
  ## `waitMs` is time to sleep if [continueDownload] is waiting for more data to read.
  ## `giveupSecs` is total time for replying before socket is closed.
  ## Continue response with calls to `continueDownload`.
  ## End response with finishDownload.
  ## 
  ## See examples/streamingdownloadtest.nim for a concrete example.
  ## 
  when defined(nimdoc): discard
  else:
    stream.waitms = waitMs
    stream.timeout = initDuration(seconds = giveupSecs)
    stream.start = getMonoTime()
    return replyStart(code, -1, ["Transfer-Encoding: chunked"]) != Fail


proc continueDownload*(chunk: string): bool =
  when defined(nimdoc): discard
  else:
    var delivered = 0
    try:
      {.gcsafe.}:
        let delimiter = fmt"{chunk.len:X}" & shortdivider
      if writeToSocket(addr delimiter, delimiter.len) == Fail: return false
    except: return false
    while true:
      if shuttingdown:
        closeSocket()
        return false
      if getMonoTime() - stream.start > stream.timeout:
        closeSocket(TimedOut, "reached giveupSecs = " & $stream.timeout)
        return false
      let (state , len) = tryWriteToSocket(addr chunk, delivered, chunk.len - delivered)
      delivered += len
      if state == Fail: return false
      elif state == TryAgain: suspend(stream.waitms)
      elif state == Complete or delivered == chunk.len:
        {.gcsafe.}:
          if writeToSocket(addr shortdivider, shortdivider.len) == Fail: return false
        return true


proc finishDownload*(): bool {.discardable.} =
  when defined(nimdoc): discard
  else:
    {.gcsafe.}:
      let delimiter = "0" & longdivider
    if writeToSocket(addr delimiter, delimiter.len) == Fail: return false
    return replyFinish() != Fail


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