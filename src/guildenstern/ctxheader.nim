## An optimized HttpCtx handler for requests that never contain a body (content-length is zero), like GET requests.
## Use either of HttpCtx's ``parseHeaders`` to parse headers, if needed.

from posix import recv

import guildenserver
import ctxhttp
export ctxhttp

var
  HeaderCtxId: CtxId
  requestlineparsing = true
  requestCallback: RequestCallback
  
  ctx {.threadvar.}: HttpCtx


{.push checks: off.}

template isFinished: bool =
  request[context.requestlen-4] == '\c' and request[context.requestlen-3] == '\l' and request[context.requestlen-2] == '\c' and request[context.requestlen-1] == '\l'

when not defined(nimdoc): # only interesting for handler writers
  proc receiveHeader*(context: HttpCtx): bool {.gcsafe, raises:[].} =
    while true:
      if shuttingdown: return false
      let ret = 
        if context.requestlen == 0: recv(context.socketdata.socket, addr request[0], MaxHeaderLength + 1, 0x40) # 0x40 = MSG_DONTWAIT
        else: recv(context.socketdata.socket, addr request[context.requestlen], MaxHeaderLength + 1, 0)
      checkRet()
      if ret == MaxHeaderLength + 1: (context.notifyError("receiveHeader: Max header size exceeded"); return false)
      context.requestlen += ret
      if isFinished: break
    return context.requestlen > 0

{.pop.}


proc handleHeaderRequest(gs: ptr GuildenServer, data: ptr SocketData) {.gcsafe, nimcall, raises: [].} =
  if ctx == nil: ctx = new HttpCtx
  if request.len < MaxRequestLength + 1: request = newString(MaxRequestLength + 1)
  initHttpCtx(ctx, gs, data)    
  if ctx.receiveHeader() and (not requestlineparsing or ctx.parseRequestLine()):
    {.gcsafe.}: requestCallback(ctx)
    

proc initHeaderCtx*(gs: var GuildenServer, onrequestcallback: proc(ctx: HttpCtx){.gcsafe, nimcall, raises: [].}, ports: openArray[int], parserequestline = true) =
  ## Initializes the headerctx handler for given ports with given request callback. By setting `parserequestline` to false this becomes a pass-through handler
  ## that does no handling for the request.
  HeaderCtxId  = gs.getCtxId()
  {.gcsafe.}: 
    requestCallback = onrequestcallback
    requestlineparsing = parserequestline
    gs.registerHandler(HeaderCtxId, handleHeaderRequest, ports)