## An optimized HttpCtx handler for requests that never contain a body (content-length is zero), like GET requests.
## Use either of HttpCtx's ``parseHeaders`` to parse headers, if needed.
## 
## **Example:**
##
## .. code-block:: Nim
##
##   import guildenstern, guildenstern/ctxheader
## 
##   const headerfields = ["user-agent", "accept-language"]
##   let htmlstart = "<!doctype html><title>.</title><body>your user-agent: "
##   let htmlmore = "<br>your accept-language: "
##   var headervalues {.threadvar.}: array[2, string]
##     
##   proc onRequest(ctx: HttpCtx) =
##     {.gcsafe.}:
##       ctx.parseHeaders(headerfields, headervalues)
##       let contentlength = htmlstart.len + headervalues[0].len  + htmlmore.len + headervalues[1].len
##       if not ctx.replyStart(Http200, contentlength, unsafeAddr htmlstart): return
##       discard ctx.replyMore(addr headervalues[0])
##       discard ctx.replyMore(unsafeAddr htmlmore)
##       ctx.replyLast(addr headervalues[1])
##     
##   var server = new GuildenServer
##   server.initHeaderCtx(onRequest, 5050)
##   echo "Point your browser to localhost:5050"
##   server.serve()

from posix import recv, SocketHandle

when not defined(nimdoc):
  import guildenstern
  export guildenstern
else:
  import guildenserver, ctxhttp

var
  requestlineparsing = true
  requestCallback: RequestCallback
  
  ctx {.threadvar.}: HttpCtx


{.push checks: off.}

template isFinished: bool =
  request[context.requestlen-4] == '\c' and request[context.requestlen-3] == '\l' and request[context.requestlen-2] == '\c' and request[context.requestlen-1] == '\l'

proc receiveHeader*(context: HttpCtx): bool {.gcsafe, raises:[].} =
  ## only useful when writing new handlers
  while true:
    if shuttingdown: return false
    let ret =
      if context.requestlen == 0: recv(posix.SocketHandle(context.socketdata.socket), addr request[0], MaxHeaderLength + 1, 0x40) # 0x40 = MSG_DONTWAIT
      else: recv(posix.SocketHandle(context.socketdata.socket), addr request[context.requestlen], MaxHeaderLength + 1, 0)
    checkRet()
    if ret == MaxHeaderLength + 1: (context.gs.notifyError("receiveHeader: Max header size exceeded"); return false)
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
    

proc initHeaderCtx*(gs: var GuildenServer, onrequestcallback: proc(ctx: HttpCtx){.gcsafe, nimcall, raises: [].}, port: int, parserequestline = true) =
  ## Initializes the headerctx handler for given ports with given request callback. By setting `parserequestline` to false this becomes a pass-through handler
  ## that does no handling for the request.
  {.gcsafe.}: 
    requestCallback = onrequestcallback
    requestlineparsing = parserequestline
    discard gs.registerHandler(handleHeaderRequest, port)