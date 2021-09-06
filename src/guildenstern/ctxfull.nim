## The go-to HttpCtx handler for most use cases, when optimal performance is not critical. Parses request line and headers automatically.
##
## **Example:**
##
## .. code-block:: Nim
##
##    import cgi
##    import guildenstern/ctxfull
##
##    proc handleGet(ctx: HttpCtx, headers: StringTableRef) =
##      let htmlstart = "<!doctype html><title>GuildenStern FullCtx Example</title><body>request-URI: "
##      let htmlmore = "<br>user-agent: "
##      let htmllast = """<br><form action="/post" method="post"><input name="say" id="say" value="Hi"><button>Send"""
##      if not headers.hasKey("user-agent"): (ctx.reply(Http412); return)
##      var uri = ctx.getUri()
##      var useragent = headers.getOrDefault("user-agent")
##      let contentlength = htmlstart.len + uri.len + htmlmore.len + useragent.len + htmllast.len
##      if not ctx.replyStart(Http200, contentlength, htmlstart): return
##      if not ctx.replyMore(uri): return
##      if not ctx.replyMore(htmlmore): return
##      if not ctx.replyMore(useragent): return
##      ctx.replyLast(htmllast)
##
##    proc handlePost(ctx: HttpCtx) =
##      var html = "<!doctype html><title>GuildenStern FullCtx Example</title><body>You said: "
##      try:
##        html.add(readData(ctx.getBody()).getOrDefault("say"))
##        ctx.reply(Http200, html)
##      except: ctx.reply(Http412)
##
##    proc onRequest(ctx: HttpCtx, headers: StringTableRef) =
##      if ctx.isMethod("POST"): ctx.handlePost()
##      else: ctx.handleGet(headers)
##
##    var server = new GuildenServer
##    server.initFullCtx(onRequest, 5050)
##    echo "Point your browser to localhost:5050/any/request-uri/path/"
##    server.serve()

from posix import recv, SocketHandle
import strtabs
export strtabs
import httpcore
export httpcore

when not defined(nimdoc):
  import guildenstern
  export guildenstern
else:
  import guildenserver, ctxhttp


type
  FullRequestCallback* = proc(ctx: HttpCtx, headers: StringTableRef){.gcsafe, nimcall, raises: [].}
  
  PortDatum = object
    port: uint16
    messagehandler: FullRequestCallback
  
var
  portdata: array[MaxHandlersPerCtx, PortDatum]
  ctx {.threadvar.}: HttpCtx
  headers {.threadvar.}: StringTableRef


proc at(port: uint16): int {.inline.} =
  while portdata[result].port != port: result += 1 

{.push hints:off.}

proc receiveHttp*(actx: HttpCtx): bool {.gcsafe, raises:[] .} =
  ## only useful when writing new handlers
  var expectedlength = MaxRequestLength + 1
  while true:
    if shuttingdown: return false
    let ret = recv(posix.SocketHandle(actx.socketdata.socket), addr request[actx.requestlen], expectedlength - actx.requestlen, 0)
    checkRet(actx)
    let previouslen = actx.requestlen
    actx.requestlen += ret

    if actx.requestlen >= MaxRequestLength:
      actx.closeSocket(ProtocolViolated, "recvHttp: Max request size exceeded")
      return false

    if actx.requestlen == expectedlength: break

    if not actx.isHeaderreceived(previouslen, actx.requestlen):
      if actx.requestlen >= MaxHeaderLength:
        actx.closeSocket(ProtocolViolated, "recvHttp: Max header size exceeded" )
        return false
      continue

    let contentlength = actx.getContentLength()
    if contentlength == 0: return true
    expectedlength = actx.bodystart + contentlength
    if actx.requestlen == expectedlength: break
  true

{.pop.}

proc handleHttpRequest(gs: ptr GuildenServer, data: ptr SocketData) {.nimcall, raises: [].} =
  if ctx == nil: ctx = new HttpCtx
  if headers == nil: headers = newStringTable()
  initHttpCtx(ctx, gs, data)
  if ctx.receiveHttp() and ctx.parseRequestLine():
    headers.clear() # slow...
    ctx.parseHeaders(headers)
    {.gcsafe.}: portdata[at(ctx.socketdata.port)].messagehandler(ctx, headers)


proc initFullCtx*(gs: var GuildenServer, onrequestcallback: FullRequestCallback, port: int) =
  ## Initializes the fullctx handler for given port with given request callback. See example above.
  var index = 0
  while index < MaxHandlersPerCtx and portdata[index].port != 0: index += 1
  if index == MaxHandlersPerCtx: raise newException(Exception, "Cannot register over " & $MaxHandlersPerCtx & " ports per FullCtx")
  portdata[index] = PortDatum(port: port.uint16, messagehandler: onrequestcallback)
  gs.registerHandler(handleHttpRequest, port, "http")