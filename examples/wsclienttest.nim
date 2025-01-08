# nim r -d:threadsafe --mm:atomicArc -d:danger wsclienttest 

import segfaults

import std/atomics
from os import sleep
import guildenstern/[websockettester, osdispatcher]

const ClientCount = 10000
const MinRoundTrips = 100000

var clients: array[1..ClientCount, WsClient]
var roundtrips, done: Atomic[int]
var closing: bool

proc doShutdown(msg: string) =
  {.gcsafe.}:
    if closing: return
    closing = true
    echo "Shutting down, because ", msg
    for client in clients: client.close(false)
    shutdown()

proc serverHandler() =
  #echo "server got message: ", getMessage()
  if closing: return
  if not wsserver.send(thesocket, "fromservertoclient"): doShutdown("Server could not reach client")

proc clientHandler(client: WsClient) =
  let r = 1 + roundtrips.fetchAdd(1)
  if r mod 10000 == 0: echo "round trips: ", r
  if r >= MinRoundTrips:
    done.atomicInc()
    return
  if closing: return
  if not client.send("fromclienttoserver"): doShutdown("Client could not reach server")


let wsServer = newWebSocketServer(nil, nil, serverHandler, nil)
wsServer.start(5050)

for i in 1..ClientCount:
  clients[i] = connect("http://127.0.0.1:5050", clientHandler, 5)
  if not clients[i].connected: quit("could not connect to server")
echo ClientCount, " clients connected"
for i in 1..ClientCount:
  if not clients[i].send("start"): quit("could not send start")
  clients[i].id = i
  # echo "Client ", i, " started"
echo ClientCount, " clients started"
  

while not shuttingdown and done.load < ClientCount: sleep(200)
doShutdown("We are done!")