import ctxhttp
export ctxhttp


var
  FullCtxId: CtxId
  requestCallback: RequestCallback
  ctx {.threadvar.}: HttpCtx 


proc receiveHttp(): bool {.gcsafe, raises:[] .} =
  var expectedlength = MaxRequestLength + 1
  while true:
    if ctx.gs.serverstate == Shuttingdown: return false
    let ret = if ctx.requestlen == 0: recv(ctx.socketdata.socket, addr request[ctx.requestlen], expectedlength - ctx.requestlen, 0x40) # MSG_DONTWAIT
      else: recv(ctx.socketdata.socket, addr request[ctx.requestlen], expectedlength - ctx.requestlen, 0)
    if ctx.gs.serverstate == Shuttingdown: return false
    if ctx.requestlen == 0 and ret == -1: return false
    checkRet()   
    let previouslen = ctx.requestlen
    ctx.requestlen += ret
    
    if ctx.requestlen >= MaxRequestLength:
      ctx.notifyError("recvHttp: Max request size exceeded")
      ctx.closeSocket()
      return false
    
    if ctx.requestlen == expectedlength: break
    
    if not ctx.isHeaderreceived(previouslen, ctx.requestlen):
      if ctx.requestlen >= MaxHeaderLength:
        ctx.notifyError("recvHttp: Max header size exceeded")
        ctx.closeSocket()
        return false
      continue
    
    let contentlength = ctx.getContentLength()
    if contentlength == 0: return true
    expectedlength = ctx.bodystart + contentlength
    if ctx.requestlen == expectedlength: break
  true


proc handleHttpRequest(gs: ptr GuildenServer, data: ptr SocketData) {.gcsafe, nimcall, raises: [].} =
  if ctx == nil: ctx = new HttpCtx
  if request.len < MaxRequestLength + 1: request = newString(MaxRequestLength + 1)
  initHttpCtx(ctx, gs, data)    
  if receiveHttp() and ctx.parseRequestLine():
    {.gcsafe.}: requestCallback(ctx)
      

proc initFullCtx*(gs: var GuildenServer, onrequestcallback: proc(ctx: HttpCtx){.gcsafe, nimcall, raises: [].}, ports: openArray[int]) =
  FullCtxId  = gs.getCtxId()
  {.gcsafe.}: 
    requestCallback = onrequestcallback
    gs.registerHandler(FullCtxId, handleHttpRequest, ports)