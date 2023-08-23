import locks
import guildenstern/[dispatcher, httpserver, websocketserver]

let html = """<!doctype html><title></title>
  <script> let websocket = new WebSocket("ws://" + location.host.slice(0, -1) + '2')
  websocket.onmessage = function(evt) {
    let element = document.createElement("li")
    element.innerHTML = evt.data
    document.getElementById("ul").appendChild(element)
  }
  async function get() {
    const response = await fetch("http://" + location.host + "/get")
    const result = await response.text()
    console.log(result)
  }
  </script>
  <body><button style="position: fixed" onclick="websocket.send('hallo')">say hallo</button>
  <button style="position: fixed; left: 100px" onclick="websocket.close()">close</button>
  <button style="position: fixed; left: 200px" onclick="get()">get</button>
  <ul id="ul"></ul>
  """

var
  lock: Lock
  wsconnections = newSeq[SocketHandle]()
  messages: int

proc onUpgradeRequest(): bool =
  {.gcsafe.}:
    echo "upgrade thread ", getThreadId(), " for socket ", http.socketdata.socket
    withLock(lock): wsconnections.add(http.socketdata.socket)
  true

proc onMessage() =
  if ws.socketdata.socket.int == 0:
    echo "zero socket"
    return
  if getRequest() == "hallo":
    for i in 1 .. 100:
      {.gcsafe.}: 
        if wsconnections.len == 0:
          echo "nobody here"
          break
        messages += 1
        let reply = $messages & " hello from thread " & $getThreadId()
        discard wsserver.send(wsconnections, reply)
  else:
    echo "ctxws failed: could not receive hallo"
    shutdown()
  
proc onLost(socketdata: ptr SocketData, cause: SocketCloseCause, msg: string) =
  echo cause, ": at thread ", getThreadId(), " with socket ", socketdata.socket
  {.gcsafe.}:
    withLock(lock): wsconnections.del(wsconnections.find(socketdata.socket))
        
proc onRequest() =
  echo "at first: ", getUri()
  if getUri() == "/get":
    let msg = "yeah " & $getThreadId()
    reply(msg)
  else: {.gcsafe.}: reply(html)

proc onRequest2() =
  echo "at second: ", getUri()
  if getUri() == "/get":
    let msg = "yeah " & $getThreadId()
    reply(msg)
  else: {.gcsafe.}: reply(html)

initLock(lock)
echo "Starting at ports 5050, 5051"
let server1 = newHttpServer(onRequest)
server1.start(5050, 8)
let server2 = newHttpServer(onRequest2)
server2.start(5051)
let wsserver = newWebsocketServer(onUpgradeRequest, nil, onMessage, onLost)
wsserver.start(5052)
joinThreads(server1.thread, server2.thread, wsserver.thread)
echo "Stopped"
deinitLock(lock)