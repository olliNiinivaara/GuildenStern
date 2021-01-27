import guildenstern/ctxheader
from times import now, format

proc doShutdown() =
  echo "Stopping ctxheader e2e test servers at ", now().format("HH:mm:ss")
  shutdown()

proc handleHttpRequest*(ctx: HttpCtx) {.gcsafe, raises: [].} =
  var body = "replyfromport" & $ctx.socketdata.port
  doAssert ctx.getMethod() == "GET"
  var headers: array[1, string]
  ctx.parseHeaders(["connection"], headers)
  assert(headers[0] == "keep-alive")
  ctx.reply(body)
  if ctx.getUri() == "/stop": doShutdown()

proc startServer(params: tuple[multithreaded: bool, port1: int, port2: int]) =
  var server = new GuildenServer
  server.initHeaderCtx(handleHttpRequest, params.port1)
  server.initHeaderCtx(handleHttpRequest, params.port2)
  server.registerTimerhandler(doShutdown, 30000)
  server.serve(params.multithreaded)
  
echo "Starting ctxheader e2e test servers at ", now().format("HH:mm:ss")
var threads: array[2, Thread[tuple[multithreaded: bool, port1: int, port2: int]]]
createThread(threads[0], startServer, (true, 8070, 8071))
createThread(threads[1], startServer, (false, 8090, 8091))
joinThreads(threads)