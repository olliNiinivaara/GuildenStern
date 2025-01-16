# nim r websockettest.nim 
# and open couple of browser tabs at localhost:5050

import locks
import guildenstern/[epolldispatcher, httpserver, websocketserver]

var
  lock: Lock
  wsconnections = newSeq[SocketHandle]()
  messages: int

proc onUpgradeRequest(): bool =
  echo "Socket ", thesocket, " requests upgrade to Websocket"
  true

proc afterUpgradeRequest() =
  {.gcsafe.}:
    withLock(lock): wsconnections.add(thesocket)
  echo "Websocket ", thesocket, " connected"

proc doShutdown() =
  withLock(lock):
    for socket in wsconnections: wsserver.sendClose(socket, 1001)
  shutdown()

proc onMessage() =
  {.gcsafe.}:
    if isMessage("shutdown"): doShutdown()
    if not isMessage("hallo"): return
    var currentconnections  = newSeq[SocketHandle]()
    for i in 1 .. 100:
      messages += 1
      let reply = $messages & " hello from thread " & $getThreadId()
      withLock(lock): currentconnections = wsconnections
      discard wsserver.send(currentconnections, reply)
  
proc onLost(Server: GuildenServer, socket: SocketHandle, cause: SocketCloseCause, msg: string) =
  echo "Websocket ", socket, " ", cause, " due to status code ", msg
  {.gcsafe.}:
    withLock(lock):
      let index = wsconnections.find(socket)
      if index != -1: wsconnections.del(index)
        
proc onRequest() =
  reply """<!doctype html><title></title>
  <script> let websocket = new WebSocket("ws://" + location.host.slice(0, -1) + '1')
  websocket.onmessage = function(evt) {
    let element = document.createElement("li")
    element.innerHTML = evt.data
    document.getElementById("ul").appendChild(element)
    window.scrollTo(0, document.body.scrollHeight)}
  websocket.onclose = function(evt) {
    document.querySelectorAll('button').forEach(b => b.style.visibility='hidden')
    alert("Websocket connection closed with code " + evt.code)}
  </script>
  <body><button style="position: fixed" onclick="websocket.send('hallo')">say hallo</button>
  <button style="position: fixed; left: 100px" onclick="websocket.close(4321)">close</button>
  <body><button style="position: fixed; left: 160px" onclick="websocket.send('shutdown')">shutdown</button>
  <ul id="ul"></ul>
  """

initLock(lock)
let server = newHttpServer(onRequest, NONE, false, NoBody)
if not server.start(5050): quit 1
let wsserver = newWebsocketServer(onUpgradeRequest, afterUpgradeRequest, onMessage, onLost, TRACE)
if not wsserver.start(5051, 2): quit 2
joinThreads(server.thread, wsserver.thread)
deinitLock(lock)