import guildenstern/[altdispatcher, httpserver]

proc handleHttpRequest*() =
  echo "sovellukseen asti päästiin"
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
var server1 = newHttpServer(handleHttpRequest, headerfields = ["connection"])
var server2 = newHttpServer(handleHttpRequest, TRACE, headerfields = ["connection"])
server1.onCloseSocketCallback = onCloseSocket
server2.onCloseSocketCallback = onCloseSocket
server1.start(8070)
server2.start(8071, 40)
joinThreads(server1.thread, server2.thread)