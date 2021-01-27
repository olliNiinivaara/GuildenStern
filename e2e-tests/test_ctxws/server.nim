import nativesockets, locks
from times import now, format
import guildenstern/[ctxws, ctxheader]
    
let html = """<!doctype html><title>WsCtx</title>
  <script>
  let websocket = new WebSocket("ws://" + location.host.slice(0, -1) + '1')
  let messagecount = 0
  websocket.onmessage = function(evt) {
    let element = document.createElement("li")
    element.id = "li" + (++messagecount)
    element.innerHTML = evt.data
    document.getElementById("ul").appendChild(element)}
  </script>
  <body><button id="wsbutton" onclick="websocket.send('hallo')">say hallo</button>
  <button id="closebutton" onclick="websocket.close()">close</button><ul id="ul">"""
  
var server = new GuildenServer
var lock: Lock
var socket = osInvalidSocket
var halloreceived: bool

proc doShutdown() =
  echo "Stopping ctxws e2e test server at ", now().format("HH:mm:ss")
  shutdown()

proc onUpgradeRequest(ctx: WsCtx): bool =
  withLock(lock): socket = ctx.socketdata.socket
  true

proc onMessage(ctx: WsCtx) =
  doAssert(ctx.getRequest() == "hallo")
  halloreceived = true
  
proc sendMessage() =
  withLock(lock):
    if socket != osInvalidSocket:
      let reply = "hello"
      discard server.sendWs(socket, reply)

proc onLost(ctx: Ctx, cause: SocketCloseCause, msg: string) =
  withLock(lock):
    if ctx.socketdata.socket.int == socket.int:
      echo cause
      socket = osInvalidSocket
      doShutdown()
        
proc onRequest(ctx: HttpCtx) = ctx.reply(Http200, html)

server.initHeaderCtx(onRequest, 5050)
server.initWsCtx(onUpgradeRequest, onMessage, 5051)
server.registerTimerhandler(sendMessage, 500)
server.registerTimerhandler(doShutdown, 30000)
server.registerConnectionclosedhandler(onLost)
echo "Starting ctxws e2e test server on port 5050 at ", now().format("HH:mm:ss")
initLock(lock)
server.serve()
deinitLock(lock)
doAssert(halloreceived)
