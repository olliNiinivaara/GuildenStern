import guildenstern/[dispatcher, httpserver]
from times import now, format
from os import sleep

   
proc handleHttpRequest*() {.gcsafe, raises: [].} =
  var body = "replyfromport" & $(server.port)
  doAssert getMethod() == "GET"
  var header: array[1, string]
  parseHeaders(["connection"], header)
  doAssert(header[0] == "keep-alive")
  doAssert(isHeader("connection", "keep-alive"))
  if getUri() == "/close":
    closeSocket(http.socketdata)
    return
  reply(body)
  if getUri() == "/stop":
    let stop = "stop"
    reply(stop)
    sleep(1000)
    shutdown()
  

#[proc waitUntilTheEnd(gs: ptr GuildenServer) {.thread.} =
  var sleeps: int
  while true:
    if shuttingdown or sleeps > 10000: break
    sleep(3000)
    sleeps += 3000
  shutdown()]#

echo "Starting e2e test servers at ", now().format("HH:mm:ss")
let server1 = newHttpServer(handleHttpRequest)
let server2 = newHttpServer(handleHttpRequest)
server1.start(8070, 1, TRACE)
server2.start(8071, 0, TRACE)
joinThreads(server1.thread, server2.thread)
echo "Stopped e2e test servers at ", now().format("HH:mm:ss")


#[

kun kutsuu close, se serveri hiljenee

]#