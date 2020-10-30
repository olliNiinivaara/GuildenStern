import guildenserver
import httpctx
export httpctx


var
  HeaderCtxId: CtxId
  requestCallback: RequestCallback
  ctx {.threadvar.}: HttpCtx 

{.push checks: off.}

template isFinished: bool =
  httprequest[ctx.requestlen-4] == '\c' and httprequest[ctx.requestlen-3] == '\l' and httprequest[ctx.requestlen-2] == '\c' and httprequest[ctx.requestlen-1] == '\l'

proc receiveHeader(): bool {.gcsafe, raises:[].} =
  while true:
    if ctx.gs.serverstate == Shuttingdown: return false
    let ret = 
      if ctx.requestlen == 0: recv(ctx.socketdata.socket, addr httprequest[0], MaxHeaderLength + 1, 0x40) # 0x40 = MSG_DONTWAIT
      else: recv(ctx.socketdata.socket, addr httprequest[ctx.requestlen], MaxHeaderLength + 1, 0)
    if ctx.gs.serverstate == Shuttingdown: return false
    checkRet()
    if ret == MaxHeaderLength + 1: (ctx.notifyError("receiveHeader: Max header size exceeded"); return false)
    ctx.requestlen += ret
    if isFinished: break
  return ctx.requestlen > 0

{.pop.}


proc handleHeaderRequest(gs: ptr GuildenServer, data: ptr SocketData) {.gcsafe, nimcall, raises: [].} =
  if ctx == nil: ctx = new HttpCtx
  if httprequest.len < MaxRequestLength + 1: httprequest = newString(MaxRequestLength + 1)
  initHttpCtx(ctx, gs, data)    
  if receiveHeader() and ctx.parseRequestLine():
    {.gcsafe.}: requestCallback(ctx)
    

proc initHeaderCtx*(gs: var GuildenServer, onrequestcallback: proc(ctx: HttpCtx){.gcsafe, nimcall, raises: [].}, ports: openArray[int]) =
  HeaderCtxId  = gs.getCtxId()
  {.gcsafe.}: 
    requestCallback = onrequestcallback
    gs.registerHandler(HeaderCtxId, handleHeaderRequest, ports)