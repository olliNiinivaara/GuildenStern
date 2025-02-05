import guildenstern/[dispatcher, httpserver]

proc handleHttpRequest*() =
  let body = "replyfromport" & $(server.port)
  doAssert getMethod() == "GET"
  doAssert http.headers.getOrDefault("connection") == "keep-alive"
  case getUri():
    of "/close":
      reply(Http204)
      closeSocket()
    of "/stop":
      let stop = "stop"
      reply("stop")
      shutdown()
    else: reply(body)

proc onCloseSocket(server: GuildenServer, socket: SocketHandle, cause: SocketCloseCause, msg: string){.gcsafe, nimcall, raises: [].} =
  echo "closing socket ", socket, " due to ", cause, " ", msg

echo "Starting at ports 8070, 8071"
var server0 = newHttpServer(handleHttpRequest, TRACE, headerfields = ["connection"])
var server1 = newHttpServer(handleHttpRequest, TRACE, headerfields = ["connection"])
server0.onCloseSocketCallback = onCloseSocket
server1.onCloseSocketCallback = onCloseSocket
if not server0.start(8070, 1): quit()
if not server1.start(8071, 40): quit()
joinThreads(server0.thread, server1.thread)