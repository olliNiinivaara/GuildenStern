# nim r -d:threadsafe -mm:atomicArc -d:danger examples/wsmulticasttest 

import segfaults

from strutils import parseEnum
from os import sleep
import guildenstern/[altdispatcher, websocketserver, websockettester]

const
  ClientCount = 100
  MessageCount = 100

let TotalMessageCount =
  # Every client sends MessageCount times either to all odds or to all evens...
  ClientCount * MessageCount * (ClientCount div 2)

type Message = enum
  imodd = "I am odd"
  imeven = "I am even"
  oddtoodds = "hello all odds"
  eventoevens = "hello all evens"

var
  clients: array[1..ClientCount, WsClient]
  oddclients = newSeq[SocketHandle]()
  evenclients = newSeq[SocketHandle]()
  threads: array[1..ClientCount, Thread[int]]
  reports, messagecount: int
  finished = false

proc doShutdown() =
  {.gcsafe.}:
    if shuttingdown: return
    echo "shutting down..."
    shutdown()
    finished = true
    
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
      of oddtoodds:  discard wsserver.send(oddclients, $oddtoodds)
      of eventoevens:  discard wsserver.send(evenclients, $eventoevens)
    except: quit(getCurrentExceptionMsg())

proc clientHandler(client: WsClient) =
  messagecount.atomicInc()
  if shuttingdown: return
  if messagecount mod 10000 == 0:
    let msg = getMessage()
    echo client.socket, " got msg: ", msg
    echo messagecount, " messages exchanged"
  if messagecount == TotalMessageCount:
    doShutdown() 

proc threadFunc(clientid: int) {.thread.} =
  let IAmOdd = clientid mod 2 == 1
  {.gcsafe.}:
    for i in 1 .. MessageCount:
      let msg = if IAmOdd: $oddtoodds else: $eventoevens
      if not clients[clientid].send(msg):
        echo("could not send to server")
        doShutdown()
        break


let wsServer = newWebSocketServer(nil, nil, serverHandler, nil, TRACE)
if not wsServer.start(5050, 1): quit()
for i in 1 .. ClientCount:
  clients[i] = connect("http://127.0.0.1:5050", clientHandler)
  if not clients[i].connected: quit("could not connect to server")
  let isodd = i mod 2 == 1
  let msg = if isodd: $imodd else: $imeven
  let nowreports = reports
  if not clients[i].send(msg):
    echo "could not report oddity to server"
    doShutdown()
    quit()
  while nowreports == reports: sleep(1)
  echo i, "/", ClientCount, " clients connected"
for i in 1 .. ClientCount: createThread(threads[i], threadFunc, i)
joinThreads(threads)
while not finished: sleep(100)