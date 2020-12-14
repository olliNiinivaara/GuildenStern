import guildenstern/ctxheader
import httpclient
       
const headerfields = ["afield", "anotherfield"]
var headers {.threadvar.}: array[2, string]
var client {.threadvar.}: HttpClient
    
proc initializeThreadvars() =
  try:
    client = newHttpClient()
    client.headers = newHttpHeaders(
      { "afield": "afieldvalue", "bfield": "bfieldvalue" })
  except: echo getCurrentExceptionMsg()
     
proc sendRequest() =
  try: discard client.request("http://localhost:5050")
  except: echo getCurrentExceptionMsg()
   
proc onRequest(ctx: HttpCtx) =
  ctx.parseHeaders(headerfields, headers)
  doAssert(headers[0] == "afieldvalue")
  ctx.reply(Http204)
  shutdown()
   
var server = new GuildenServer
server.registerThreadInitializer(initializeThreadvars)
server.initHeaderCtx(onRequest, 5050)
server.registerTimerhandler(sendRequest, 1000)
server.serve()