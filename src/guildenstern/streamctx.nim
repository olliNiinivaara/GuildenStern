import guildenserver, httpctx
export httpctx


type
  StreamCtx* = ref object of HttpCtx
    contentlength*: int64
    contentreceived*: int64
    contentdelivered*: int64

  RequestCallback = proc(ctx: StreamCtx){.gcsafe, raises: [].}


var
  StreamCtxId: CtxId
  requestCallback: RequestCallback
  ctx {.threadvar.}: StreamCtx 


proc receiveHeader(): bool {.gcsafe, raises:[].} =
  while true:
    if ctx.gs.serverstate == Shuttingdown: return false
    let ret = recv(ctx.socketdata.socket, addr httprequest[ctx.requestlen], MaxHeaderLength + 1, 0)
    if ctx.gs.serverstate == Shuttingdown: return false
    echo ret
    if ret < 1:
      ctx.closeSocket()
      return false
    if ret > MaxHeaderLength: (ctx.notifyError("receiveHeader: Max header size exceeded"); return false)
    ctx.requestlen += ret
    if ctx.isHeaderreceived(ctx.requestlen - ret, ctx.requestlen): break
  ctx.contentlength = ctx.getContentLength()
  ctx.contentreceived = ctx.requestlen - ctx.bodystart
  true


proc hasData*(ctx: StreamCtx): bool  =
  return ctx.contentlength > 0 and ctx.contentdelivered < ctx.contentlength


proc receiveChunk*(ctx: StreamCtx): int {.gcsafe, raises:[] .} =
  if ctx.gs.serverstate == Shuttingdown: return -1
  
  if ctx.contentdelivered == 0 and ctx.contentreceived > 0:
    httprequest = httprequest[ctx.bodystart ..< ctx.requestlen]
    ctx.contentdelivered = ctx.requestlen - ctx.bodystart
    ctx.requestlen = ctx.contentdelivered.int
    return ctx.contentdelivered.int

  let ret = recv(ctx.socketdata.socket, addr httprequest[0], (ctx.contentlength - ctx.contentreceived).int, 0)    
  if ctx.gs.serverstate == Shuttingdown: return -1      
  if ret < 1:
    if ret == -1:
      let lastError = osLastError().int
      if lastError != 2 and lastError != 9 and lastError != 32 and lastError != 104:
        ctx.notifyError("socket error: " & $lastError & " " & osErrorMsg(OSErrorCode(lastError)))
      ctx.closeSocket()
      return -1
  ctx.contentreceived += ret
  ctx.contentdelivered += ret
  ctx.requestlen = ret
  return ctx.requestlen


proc handleHeaderRequest(gs: ptr GuildenServer, data: ptr SocketData) {.gcsafe, nimcall, raises: [].} =
  if ctx == nil: ctx = new StreamCtx
  if httprequest.len < MaxRequestLength + 1: httprequest = newString(MaxRequestLength + 1)
  initHttpCtx(ctx, gs, data)
  ctx.contentlength = -1
  ctx.contentreceived = 0
  ctx.contentdelivered = 0
  if receiveHeader() and ctx.parseRequestLine():
    {.gcsafe.}: requestCallback(ctx)
    

proc initStreamCtx*(gs: var GuildenServer, onrequestcallback: proc(ctx: StreamCtx){.gcsafe, nimcall, raises: [].}, ports: openArray[int]) =
  StreamCtxId  = gs.getCtxId()
  {.gcsafe.}: 
    requestCallback = onrequestcallback
    gs.registerHandler(StreamCtxId, handleHeaderRequest, ports)