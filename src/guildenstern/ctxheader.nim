## An optimized HttpCtx handler for requests that never contain a body (content-length is zero), like GET requests.
## Use either of HttpCtx's ``parseHeaders`` to parse headers, if needed.
## 
## **Example:**
##
## .. code-block:: Nim
##
##    import guildenstern/ctxheader
##       
##    const headerfields = ["user-agent", "accept-language"]
##    var headervalues {.threadvar.}: array[2, string]
##        
##    proc onRequest(ctx: HttpCtx) =
##      let htmlstart = "<!doctype html><title>.</title><body>your user-agent: "
##      let htmlmore = "<br>your accept-language: "
##      ctx.parseHeaders(headerfields, headervalues)
##      let contentlength = htmlstart.len + headervalues[0].len  + htmlmore.len + headervalues[1].len
##      if not ctx.replyStart(Http200, contentlength, htmlstart): return
##      if not ctx.replyMore(headervalues[0]): return
##      if not ctx.replyMore(htmlmore): return
##      ctx.replyLast(headervalues[1])
##        
##    var server = new GuildenServer
##    server.initHeaderCtx(onRequest, 5050)
##    echo "Point your browser to localhost:5050"
##    server.serve()

from posix import recv, SocketHandle

when not defined(nimdoc):
  import guildenstern
  export guildenstern
else:
  import guildenserver, ctxhttp

type PortDatum = object
  port: uint16
  messagehandler: proc(ctx: HttpCtx) {.gcsafe, nimcall, raises: [].}
  requestlineparsing: bool

var
  portdata: array[MaxHandlersPerCtx, PortDatum]
  ctx {.threadvar.}: HttpCtx


{.push checks: off.}

proc at(port: uint16): int {.inline.} =
  while portdata[result].port != port: result += 1

proc receiveHeader*(actx: HttpCtx): bool {.gcsafe, raises:[].} =
  ## only useful when writing new handlers
  while true:
    if shuttingdown: return false
    let ret = recv(actx.socketdata.socket, addr request[actx.requestlen], 1 + MaxHeaderLength - actx.requestlen, 0)
    checkRet(actx)
    actx.requestlen += ret
    if actx.requestlen > MaxHeaderLength:
      actx.closeSocket(ProtocolViolated, "receiveHeader: Max header size exceeded")
      return false
    if request[actx.requestlen-4] == '\c' and request[actx.requestlen-3] == '\l' and
     request[actx.requestlen-2] == '\c' and request[actx.requestlen-1] == '\l': break
  return actx.requestlen > 0

{.pop.}

proc handleHeaderRequest(gs: ptr GuildenServer, data: ptr SocketData) {.gcsafe, nimcall, raises: [].} =
  if ctx == nil: ctx = new HttpCtx
  initHttpCtx(ctx, gs, data)
  let datum = portdata[at(ctx.socketdata.port)]    
  if ctx.receiveHeader() and (not datum.requestlineparsing or ctx.parseRequestLine()):
    {.gcsafe.}: datum.messagehandler(ctx)
    

proc initHeaderCtx*(gs: var GuildenServer,
 onrequestcallback: proc(ctx: HttpCtx){.gcsafe, nimcall, raises: [].}, port: int, parserequestline = true) =
  ## Initializes the headerctx handler for given port with given request callback.
  ## By setting global `parserequestline` to false all HeaderCtxs become pass-through handlers
  ## that do no handling for the request.
  var index = 0
  while index < MaxHandlersPerCtx and portdata[index].port != 0: index += 1
  if index == MaxHandlersPerCtx: raise newException(Exception, "Cannot register over " & $MaxHandlersPerCtx & " ports per HeaderCtx")
  portdata[index] = PortDatum(port: port.uint16, messagehandler: onrequestcallback, requestlineparsing: parserequestline)
  gs.registerHandler(handleHeaderRequest, port, "http")