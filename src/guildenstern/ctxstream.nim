## HttpCtx handler for cases when request body must be processed as it's being received (for example, when client is uploading data to filesystem).
## 
## **Example:**
##
## .. code-block:: Nim
##
##    import guildenstern/ctxstream
##    from strutils import startsWith, find
##    
##    proc handleGet(ctx: StreamCtx) =
##      while ctx.hasData(): discard ctx.receiveChunk()
##      
##    proc handleUpload(ctx: StreamCtx) =
##      const contenttype = ["content-type"]
##      var value: array[1, string]
##      ctx.parseHeaders(contenttype, value)
##      if not value[0].startsWith("multipart/form-data; boundary="): return
##      let boundary = "--" & value[0][30 ..< value[0].high]
## 
##      var resultfile: string
##      var filestate = -1 # -1 = before file, 0 = in file, 1 = after file
##      while ctx.hasData():
##        discard ctx.receiveChunk()
##        if filestate == 1: continue
##        var filestart = 0
##        if filestate == -1:
##          filestart = request.find("\c\L\c\L") + 4
##          if filestart == 3: continue else: filestate = 0
##        var fileend = request.find(boundary, filestart) - 2
##        if fileend == -3: fileend = ctx.requestlen else: filestate = 1
##        resultfile.add(request[filestart ..< fileend])
##      if resultfile != "": echo resultfile
##        
##    proc onRequest(ctx: StreamCtx) =
##      let html = """<!doctype html><title>StreamCtx</title><body>
##      <form action="/upload" method="post" enctype="multipart/form-data">
##      <input type="file" id="file" name="file">
##      <input type="submit">"""
##      if ctx.startsUri("/upload"): ctx.handleUpload() else: ctx.handleGet()
##      ctx.reply(Http200, html)
##    
##    var server = new GuildenServer
##    server.initStreamCtx(onRequest, 5050)
##    echo "Point your browser to localhost:5050"
##    server.serve()

from os import osLastError, osErrorMsg, OSErrorCode
from posix import recv, SocketHandle

when not defined(nimdoc):
  import guildenstern
  export guildenstern
else:
  import guildenserver, ctxhttp


type
  StreamCtx* = ref object of HttpCtx
    contentlength*: int64
    contentreceived*: int64
    contentdelivered*: int64

  RequestCallback = proc(ctx: StreamCtx){.gcsafe, raises: [].}


var
  requestCallback: RequestCallback
  ctx {.threadvar.}: StreamCtx 


proc receiveHeader(): bool {.gcsafe, raises:[].} =
  while true:
    if shuttingdown: return false
    let ret = recv(posix.SocketHandle(ctx.socketdata.socket), addr request[ctx.requestlen], MaxHeaderLength + 1, 0)
    checkRet()
    if ret > MaxHeaderLength:
      ctx.closeSocket(ProtocolViolated, "stream receiveHeader: Max header size exceeded")
      return false
    ctx.requestlen += ret
    if ctx.isHeaderreceived(ctx.requestlen - ret, ctx.requestlen): break
  ctx.contentlength = ctx.getContentLength()
  ctx.contentreceived = ctx.requestlen - ctx.bodystart
  true


proc hasData*(ctx: StreamCtx): bool  =
  return ctx.contentlength > 0 and ctx.contentdelivered < ctx.contentlength


template checkChunckRet() =
  if shuttingdown: return -1
  if ret < 1:
    if ret == -1:
      let lastError = osLastError().int
      let cause =
        if lasterror in [2,9]: AlreadyClosed
        elif lasterror == 32: ConnectionLost
        elif lasterror == 104: ClosedbyClient
        else: NetErrored
      ctx.closeSocket(cause, osErrorMsg(OSErrorCode(lastError)))
    else: ctx.closeSocket(ClosedbyClient) # ret == 0      
    return -1


proc receiveChunk*(ctx: StreamCtx): int {.gcsafe, raises:[] .} =
  if shuttingdown: return -1
  if ctx.contentdelivered == 0 and ctx.contentreceived > 0:
    request = request[ctx.bodystart ..< ctx.requestlen]
    ctx.contentdelivered = ctx.requestlen - ctx.bodystart
    ctx.requestlen = ctx.contentdelivered.int
    return ctx.contentdelivered.int
  let ret = recv(posix.SocketHandle(ctx.socketdata.socket), addr request[0], (ctx.contentlength - ctx.contentreceived).int, 0)        
  checkChunckRet()
  ctx.contentreceived += ret
  ctx.contentdelivered += ret
  ctx.requestlen = ret
  return ctx.requestlen


proc handleHeaderRequest(gs: ptr GuildenServer, data: ptr SocketData) {.gcsafe, nimcall, raises: [].} =
  if ctx == nil: ctx = new StreamCtx
  if request.len < MaxRequestLength + 1: request = newString(MaxRequestLength + 1)
  initHttpCtx(ctx, gs, data)
  ctx.contentlength = -1
  ctx.contentreceived = 0
  ctx.contentdelivered = 0
  if receiveHeader() and ctx.parseRequestLine():
    {.gcsafe.}: requestCallback(ctx)
    

proc initStreamCtx*(gs: var GuildenServer, onrequestcallback: proc(ctx: StreamCtx){.gcsafe, nimcall, raises: [].}, port: int) =
  {.gcsafe.}: 
    requestCallback = onrequestcallback
    discard gs.registerHandler(handleHeaderRequest, port)