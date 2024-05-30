import locks
import guildenstern/[dispatcher, httpserver, websocketserver]

var
  lock: Lock
  wsconnections = newSeq[SocketHandle]()
  messages: int

proc onUpgradeRequest(): bool =
  echo "upgrade request for socket ", ws.socketdata.socket  
  true

proc afterUpgradeRequest() =
  {.gcsafe.}:
    withLock(lock): wsconnections.add(ws.socketdata.socket)
  echo "registered  websocket ", ws.socketdata.socket

proc onMessage() =
  if not isMessage("hallo"): return
  var currentconnections  = newSeq[SocketHandle]()
  for i in 1 .. 100:
    {.gcsafe.}:
      messages += 1
      let reply = $messages & " hello from thread " & $getThreadId()
      withLock(lock): currentconnections = wsconnections
    discard wsserver.send(currentconnections, reply)
  
proc onLost(socketdata: ptr SocketData, cause: SocketCloseCause, msg: string) =
  echo cause, ": socket ", socketdata.socket
  {.gcsafe.}:
    withLock(lock):
      let index = wsconnections.find(socketdata.socket)
      if index != -1: wsconnections.del(index)
        
proc onRequest() =
  let html = """<!doctype html><title></title>
  <script> let websocket = new WebSocket("ws://" + location.host.slice(0, -1) + '1')
  websocket.onmessage = function(evt) {
    let element = document.createElement("li")
    element.innerHTML = evt.data
    document.getElementById("ul").appendChild(element)
    window.scrollTo(0, document.body.scrollHeight)}
  </script>
  <body><button style="position: fixed" onclick="websocket.send('hallo')">say hallo</button>
  <button style="position: fixed; left: 100px" onclick="
    websocket.close();
    document.querySelectorAll('button').forEach(b => b.style.visibility='hidden')
  ">close</button>
  <ul id="ul"></ul>
  """
  reply(html)

initLock(lock)
let server = newHttpServer(onRequest, NONE, false, NoBody)
server.start(5050)
let wsserver = newWebsocketServer(onUpgradeRequest, afterUpgradeRequest, onMessage, onLost)
wsserver.start(5051)
joinThreads(server.thread, wsserver.thread)
deinitLock(lock)