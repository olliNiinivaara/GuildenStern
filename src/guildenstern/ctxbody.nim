## Parses headers only on demand (like HttpHeader), but also reads the request body (like CtxFull).
## Well suited for handling POST requests.
##
## **Example:**
##
## .. code-block:: Nim
##
##    import cgi, strtabs, httpcore
##    import guildenstern/ctxbody
## 
##    const headerfields = ["origin"]
##
##    proc handlePost(ctx: HttpCtx) =
##      var headervalues: array[1, string]
##      ctx.parseHeaders(headerfields, headervalues)
##      if headervalues[0] == "": ctx.reply(Http403) 
##      var html = """<!doctype html><meta charset="utf-8"><title>BodyCtx</title><body>You said """
##      try: html.add(readData(ctx.getBody()).getOrDefault("say"))
##      except: (ctx.reply(Http412) ; return)
##      html.add(" from " & headervalues[0])
##      ctx.reply(Http200, html)
## 
##    proc handleGet(ctx: HttpCtx) =
##      let html = """<!doctype html><meta charset="utf-8"><title>BodyCtx</title><body>
##      <form action="/post" method="post"><input name="say" id="say" value="Hi"><button>Send"""
##      ctx.reply(html)
##
##    proc onRequest(ctx: HttpCtx) =
##      if ctx.isMethod("POST"): ctx.handlePost()
##      else: ctx.handleGet()
##
##    var server = new GuildenServer
##    server.initBodyCtx(onRequest, 5050)
##    echo "Point your browser to localhost:5050"
##    server.serve()
## 
## 

when not defined(nimdoc):
  import guildenstern
  export guildenstern
else:
  import guildenserver, ctxhttp

from ctxfull import receiveHttp


type PortDatum = object
  port: uint16
  messagehandler: proc(ctx: HttpCtx) {.gcsafe, nimcall, raises: [].}

var
  portdata: array[MaxHandlersPerCtx, PortDatum]
  ctx {.threadvar.}: HttpCtx

{.push checks: off.}

proc at(port: uint16): int {.inline.} =
  while portdata[result].port != port: result += 1

proc handleBodyRequest(gs: ptr GuildenServer, data: ptr SocketData) {.gcsafe, nimcall, raises: [].} =
  if ctx == nil: ctx = new HttpCtx
  initHttpCtx(ctx, gs, data)
  if ctx.receiveHttp() and ctx.parseRequestLine():
    {.gcsafe.}: portdata[at(ctx.socketdata.port)].messagehandler(ctx)
    
proc initBodyCtx*(gs: var GuildenServer, onrequestcallback: proc(ctx: HttpCtx){.gcsafe, nimcall, raises: [].}, port: int) =
  ## Initializes the bodyctx handler for given port with given request callback.
  var index = 0
  while index < MaxHandlersPerCtx and portdata[index].port != 0: index += 1
  if index == MaxHandlersPerCtx: raise newException(Exception, "Cannot register over " & $MaxHandlersPerCtx & " ports per BodyCtx")
  portdata[index] = PortDatum(port: port.uint16, messagehandler: onrequestcallback)
  gs.registerHandler(handleBodyRequest, port, "http")

#[ when Nim views available (currently "broken beyond repair"):
proc getBodyView*(): openArray[char] =
  ## Creates a zero-copy view to body
  # if ctx.bodystart < 1: return toOpenArray("", 0, 0)
  toOpenArray(request, ctx.bodystart, ctx.requestlen - 1) ]#

{.pop.}