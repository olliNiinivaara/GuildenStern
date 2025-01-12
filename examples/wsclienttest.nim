# nim r -d:threadsafe --mm:atomicArc -d:useMalloc -d:release wsclienttestcopy

import segfaults
import std/atomics
from os import sleep
import guildenstern/osdispatcher
import guildenstern/websocketserver
import guildenstern/websocketclient

const ClientCount = 1000
const MinRoundTrips = 100000
var roundtrips: Atomic[int]
var closing: bool
var clientele: WebsocketClientele


proc serverReceive() =
  if closing: return
  if not wsserver.send(thesocket, "fromservertoclient"):
    closeSocket(CloseCalled, "Server could not reach client")

proc doShutdown(msg: string) =
  {.gcsafe.}:
    echo "Shutting down, because ", msg
    sleep(2000)
    for client in clientele.connectedClients():
      # echo "closing ", client.id
      client.close(false)
    shutdown()
    echo "Total round trips: ", roundtrips.load

proc clientReceive(client: WebsocketClient) =
  let r = 1 + roundtrips.fetchAdd(1)
  if r mod 1000 == 0: echo "round trips: ", r
  if r >= MinRoundTrips: closing = true
  if closing: return
  if not client.send("fromclienttoserver"):
    echo("Client could not reach server")
    client.close()

proc start() =
  for i in 1..ClientCount:
    let client = clientele.newWebsocketClient("http://127.0.0.1:5050", clientReceive)
    if not client.connect(): quit("could not connect to server")
  echo ClientCount, " clients connected"
  for client in clientele.connectedClients():
    if not client.send("start"): quit("could not send start")
    # echo "Client ", client.id, " started"
  echo "All ", ClientCount, " clients started"


let wsServer = newWebSocketServer(nil, nil, serverReceive, nil, INFO)
if not wsServer.start(5050): quit 1
clientele = newWebsocketClientele(loglevel = INFO)
if not clientele.start(): quit 2
start()
while not closing and not shuttingdown: sleep(200)
doShutdown("We are done!")