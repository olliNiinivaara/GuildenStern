## Server for cases when request body must be processed as it's being received (for example, when client is uploading data to filesystem). 

import std/[times, monotimes]
from os import osLastError, osErrorMsg, OSErrorCode, sleep
from posix import recv, SocketHandle, EAGAIN, EWOULDBLOCK
from strutils import find, startsWith


import httpserver
export httpserver


type
  StreamingHandler* = ref object of HttpHandler
    contentlength*: int64
    contentreceived*: int64
    contentdelivered*: int64
    waitms: int
    timeout: Duration
    start: MonoTime
    boundary: string
    filestate: int
  
  StreamingServer* = ref object of GuildenServer
    requestCallback*: proc(){.gcsafe, nimcall, raises: [].}

  StreamState* = enum
    Fail = -1
    TryAgain = 0
    Success = 1
    Complete = 2
  

proc isStreamingHandler*(): bool = return guildenhandler is StreamingHandler

template stream*(): untyped = StreamingHandler(guildenhandler)


proc receiveHeader(): bool {.gcsafe, raises:[].} =
  while true:
    if shuttingdown: return false
    let ret = recv(http.socketdata.socket, addr http.request[http.requestlen], 1 + MaxHeaderLength - http.requestlen, 0)
    checkRet()
    http.requestlen += ret
    if isHeaderreceived(http.requestlen - ret, http.requestlen): break
    if http.requestlen > MaxHeaderLength:
      server.doCloseSocket(http.socketdata, ProtocolViolated, "stream receiveHeader: Max header size exceeded")
      return false
  stream.contentlength = getContentLength()
  stream.contentreceived = http.requestlen - http.bodystart
  true


proc hasData*(): bool  =
  return stream.contentlength > 0 and stream.contentdelivered < stream.contentlength


template checkChunckRet() =
  if shuttingdown: return -1
  if ret < 1:
    if ret == -1:
      let lastError = osLastError().int
      let cause =
        if lasterror in [2,9]: AlreadyClosed
        elif lasterror in [11, EAGAIN.int, EWOULDBLOCK.int]: TimedOut
        elif lasterror == 32: ConnectionLost
        elif lasterror == 104: ClosedbyClient
        else: NetErrored
      server.doCloseSocket(http.socketdata, cause, osErrorMsg(OSErrorCode(lastError)))
    else: server.doCloseSocket(http.socketdata, ClosedbyClient, "") # ret == 0      
    return -1


const DONTWAIT = 0x40.cint

proc receiveChunk*(): int {.gcsafe, raises:[] .} =
  if shuttingdown: return -1
  if stream.contentdelivered == 0 and stream.contentreceived > 0:
    stream.contentdelivered = stream.requestlen - stream.bodystart
    stream.requestlen = stream.contentdelivered.int
    return stream.contentdelivered.int
  let ret = recv(stream.socketdata.socket, addr stream.request[0], (stream.contentlength - stream.contentreceived).int, DONTWAIT)
  if ret == -1 and osLastError().int in [EAGAIN.int, EWOULDBLOCK.int]: return 0
  checkChunckRet()
  stream.contentreceived += ret
  stream.contentdelivered += ret
  stream.requestlen = ret
  return stream.requestlen


proc startReceiveMultipart*(waitMs = 100, giveupSecs = 10): bool {.gcsafe, raises: [].} =
  const contenttype = ["content-type"]
  var value: array[1, string]
  parseHeaders(contenttype, value)
  if not value[0].startsWith("multipart/form-data; boundary="): return false
  stream.waitms = waitMs
  stream.timeout = initDuration(seconds = giveupSecs)
  stream.start = getMonoTime()
  stream.filestate = -1
  stream.boundary = "--" & value[0][30 .. value[0].high] & "--"
  return true


proc receiveMultipart*(): (StreamState , string) {.gcsafe, raises: [].} =
  if not hasData(): return (Complete , "")
  let received = receiveChunk()
  if received == -1: return (Fail , "")
  if received == 0:
    sleep(stream.waitms)
    if getMonoTime() - stream.start > stream.timeout: return (Fail, "")
    return (TryAgain , "")
  if stream.filestate == 1:
    while receiveChunk() > 0: continue
    return (Complete , "")

  var filestart = 0
  if stream.filestate == -1:
    filestart = stream.request.find("\c\L\c\L") + 4
    if filestart == 3:
      while filestart == 3 and receiveChunk() > 0:
        filestart = stream.request.find("\c\L\c\L") + 4
      if filestart == 3: return (Fail , "")
  stream.filestate = 0
  var fileend = stream.request.find(stream.boundary, filestart) - 2
  if fileend == -3: fileend = stream.requestlen
  return (Success , stream.request[filestart .. fileend - filestart])


proc handleStreamRequest(data: ptr SocketData) {.gcsafe, nimcall, raises: [].} =
  if (unlikely)guildenhandler == nil: guildenhandler = new StreamingHandler
  prepareHttpHandler(data)
  stream.contentlength = -1
  stream.contentreceived = 0
  stream.contentdelivered = 0
  if receiveHeader() and parseRequestLine():
    {.gcsafe.}: StreamingServer(data.server).requestCallback()
    

proc newStreamingServer*(onrequestcallback: proc(){.gcsafe, nimcall, raises: [].}): StreamingServer =
  result = new StreamingServer
  result.registerHandler(handleStreamRequest)
  result.requestCallback = onrequestcallback