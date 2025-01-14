# nim r --mm:atomicArc -d:release wsclienttestcopy

# import segfaults

import std/atomics
from os import sleep
import guildenstern/[dispatcher, websocketserver, websocketclient]

const ClientCount = 4000 # ulimit -n something more first
const MinRoundTrips = 100000
var roundtrips: Atomic[int]
var clientele: WebsocketClientele
var closing: bool


proc serverReceive() =
  if not wsserver.send(thesocket, "fromservertoclient"):
    closeSocket(CloseCalled, "Server could not reach client")

proc doShutdown(msg: string) =
  {.gcsafe.}:
    if closing: return
    closing = true
    echo "Shutting down, because ", msg, "..."
    sleep(2000)
    for client in clientele.connectedClients():
      # echo "closing ", client.id
      client.close(true)
    shutdown()
    echo "Total round trips: ", roundtrips.load

proc clientReceive(client: WebsocketClient) =
  let r = 1 + roundtrips.fetchAdd(1)
  if r mod 1000 == 0: echo "round trips: ", r
  if r >= MinRoundTrips:
    doShutdown("We are done!")
    return
  if not client.send("fromclienttoserver"):
    echo("Client could not reach server")
    client.close()

proc start() =
  for i in 1..ClientCount:
    let client = clientele.newWebsocketClient("http://127.0.0.1:5050", clientReceive)
    if not client.connect(): quit("could not connect to server")
    if i mod 100 == 0: echo i, " clients connected"
  echo "All ", ClientCount, " clients connected"
  for client in clientele.connectedClients():
    if not client.send("start"):
      doShutdown("Could not start client " & $client.id)
      break
    if client.id mod 100 == 0: echo client.id, " clients started"
  echo "All ", ClientCount, " clients started"


let wsServer = newWebSocketServer(nil, nil, serverReceive)
if not wsServer.start(5050, 10): quit 1
clientele = newWebsocketClientele(bufferlength = 20)
if not clientele.start(threadpoolsize = 10): quit 2
start()
joinThread(wsServer.thread)