import guildenstern/[dispatcher, httpserver]

proc handleHttpRequest*() =
  let body = "replyfromport" & $(server.port)
  doAssert getMethod() == "GET"
  doAssert http.headers.getOrDefault("connection") == "keep-alive"
  case getUri():
    of "/close": closeSocket()
    of "/stop":
      let stop = "stop"
      reply(stop)
      shutdown()
    else: reply(body)

echo "Starting at ports 8070, 8071"
var server1 = newHttpServer(handleHttpRequest, headerfields = ["connection"])
var server2 = newHttpServer(handleHttpRequest, TRACE, headerfields = ["connection"])
server1.start(8070, 1)
server2.start(8071)
joinThreads(server1.thread, server2.thread)