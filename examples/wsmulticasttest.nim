# nim r -mm:atomicArc -d:danger examples/wsmulticasttest 

import segfaults

from strutils import parseEnum
from os import sleep
import guildenstern/[osdispatcher, websocketserver, websocketclient]

type Message = enum
  imodd = "I am odd"
  imeven = "I am even"
  oddtoodds = "hello all odds"
  eventoevens = "hello all evens"

const
  ClientCount = 100
  MessageCount = 100

let TotalMessageCount =
  # Every client sends MessageCount times either to all odds or to all evens...
  ClientCount * MessageCount * (ClientCount div 2)

var
  oddclients = newSeq[SocketHandle]()
  evenclients = newSeq[SocketHandle]()
  threads: array[1..ClientCount, Thread[int]]
  reports, messagecount: int
  clientele: WebsocketClientele

proc doShutdown() =
  {.gcsafe.}:
    echo "Shutting down...."
    sleep(1000)
    for client in clientele.connectedClients(): client.close()
    shutdown()
    
proc serverHandler() =
  {.gcsafe.}:
    try:
      let msg = parseEnum[Message](getMessage())
      case msg:
      of imodd:
        oddclients.add(thesocket)
        reports.atomicInc()
      of imeven:
        evenclients.add(thesocket)
        reports.atomicInc()
      of oddtoodds: discard wsserver.send(oddclients, $oddtoodds)
      of eventoevens: discard wsserver.send(evenclients, $eventoevens)
    except: doShutdown()

proc clientHandler(client: WebsocketClient) =
  messagecount.atomicInc()
  if shuttingdown: return
  if messagecount mod 10000 == 0:
    let msg = getMessage()
    #echo client.socket, " got msg: ", msg
    echo messagecount, " messages exchanged"
  if messagecount == TotalMessageCount: doShutdown() 

proc threadFunc(clientid: int) {.thread.} =
  let IAmOdd = clientid mod 2 == 1
  {.gcsafe.}:
    for i in 1 .. MessageCount:
      let msg = if IAmOdd: $oddtoodds else: $eventoevens
      if not clientele.clients[clientid].send(msg):
        echo("could not send to server")
        doShutdown()
        break

let wsServer = newWebSocketServer(nil, nil, serverHandler)
if not wsServer.start(5050, 10): quit()
clientele = newWebsocketClientele(loglevel = INFO)
if not clientele.start(): quit()
for i in 1 .. ClientCount:
  let client = clientele.newWebsocketClient("http://127.0.0.1:5050", clientHandler)
  if not client.connect(): quit("could not connect to server")
  let isodd = i mod 2 == 1
  let msg = if isodd: $imodd else: $imeven
  if not client.send(msg):
    echo "could not report oddity to server"
    doShutdown()
  echo i, "/", ClientCount, " clients connected"
while reports < ClientCount: sleep(10)
for i in 1 .. ClientCount: createThread(threads[i], threadFunc, i)
joinThreads(threads)
joinThread(wsserver.thread)