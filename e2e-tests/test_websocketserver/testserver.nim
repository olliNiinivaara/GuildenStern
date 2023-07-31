import nativesockets, locks
from times import now, format
from os import sleep
import guildenstern/[dispatcher, websocketserver]
      

var lock: Lock
# var thesocket = INVALID_SOCKET
# var halloreceived: bool


proc onUpgradeRequest(): bool =
  # withLock(lock): thesocket = gs.socketdata.socket
  true

proc sendMessage() =
  let reply = "hello"
  withLock(lock):
     {.gcsafe.}: discard wsserver.send(ws.socketdata.socket, reply)


proc onMessage() =
  if getRequest() == "hallo":
    echo "hallo received ", getThreadId()
    # sleep(2000)
    sendMessage()
  else:
    echo "ctxws failed: could not receive hallo"
    shutdown()
  

proc onLost(socketdata: ptr SocketData, cause: SocketCloseCause, msg: string) =
  withLock(lock):
      echo cause
      # shutdown()
        
proc onRequest() =
  let htmlfirst = """<!doctype html><title>WsCtx</title>"""
  let htmlsecond = """<script>
  let websocket = new WebSocket("ws://" + location.host.slice(0, -1) + '1')
  let messagecount = 0
  websocket.onmessage = function(evt) {
    let element = document.createElement("li")
    element.id = "li" + (++messagecount)
    element.innerHTML = evt.data
    document.getElementById("ul").appendChild(element)}
  </script>"""
  let htmlthird = """<body><button id="wsbutton" onclick="websocket.send('hallo')">say hallo</button>
  <button id="closebutton" onclick="websocket.close()">close</button><ul id="ul">"""
  if replyStart(Http200, htmlfirst.len + htmlsecond.len + htmlthird.len) == Fail: return
  var start = 0
  var currentchunk = htmlfirst
  while true:
    let (state, sent) = replyMore(addr currentchunk, start, currentchunk.len - start)
    case state
      of Fail: return
      of TryAgain: continue
      of Progress: start += sent
      of Complete:
        if currentchunk == htmlfirst: currentchunk = htmlsecond
        elif currentchunk == htmlsecond: currentchunk = htmlthird
        else: break
        start = 0
  replyFinish()


echo "Starting test servers at ", now().format("HH:mm:ss")
let htmlserver = newHttpServer(onRequest)
let jsonserver = newWebsocketServer(onUpgradeRequest, nil, onMessage, onLost)
htmlserver.start(5050)
jsonserver.start(5051)
joinThreads(htmlserver.thread, jsonserver.thread)
echo "Stopped test servers at ", now().format("HH:mm:ss")

