import guildenstern/[dispatcher, httpserver]

proc handleHttpRequest*() {.gcsafe, raises: [].} =
  var body = "replyfromport" & $(server.port)
  doAssert getMethod() == "GET"
  var header: array[1, string]
  parseHeaders(["connection"], header)
  doAssert(header[0] == "keep-alive")
  doAssert(isHeader("connection", "keep-alive"))
  if getUri() == "/close":
    closeSocket()
    return
  reply(body)
  if getUri() == "/stop":
    let stop = "stop"
    reply(stop)
    shutdown()

echo "Starting at ports 8070, 8071"
let server1 = newHttpServer(handleHttpRequest, TRACE)
let server2 = newHttpServer(handleHttpRequest, TRACE)
server1.start(8070, 1)
server2.start(8071, 0)
joinThreads(server1.thread, server2.thread)
echo "Stopped"