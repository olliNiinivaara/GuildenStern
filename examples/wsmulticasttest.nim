# nim r -mm:atomicArc -d:danger examples/wsmulticasttest 

import segfaults

from strutils import parseEnum
from os import sleep
import atomics
import random
import guildenstern/[dispatcher, websocketserver, websocketclient]

type Message = enum
  imodd = "I am odd"
  imeven = "I am even"
  oddtoodds = "hello all odds"
  eventoevens = "hello all evens"

const
  ClientCount = 200
  ThreadCount = 50
  MessageCount = 50

doAssert(ClientCount mod 2 == 0)

let TotalMessageCount =
  # Every Thread sends MessageCount times either to all odds or to all evens...
  ThreadCount * MessageCount * (ClientCount div 2)

var
  oddclients = newSeq[SocketHandle]()
  evenclients = newSeq[SocketHandle]()
  threads: array[1..ThreadCount, Thread[void]]
  reports, sendcount, receivecount: Atomic[int]
  closing: bool
  clientele: WebsocketClientele


proc doShutdown() =
  {.gcsafe.}:
    if closing: return
    closing = true
    echo "Shutting down now"
    for client in clientele.connectedClients():
      # echo "Closing client ", client.id
      client.close(true)
    shutdown()
    

proc serverHandler() =
  {.gcsafe.}:
    if closing: return
    sleep(1) # lets cheat a little
    var success: bool
    var msg: Message
    try:
      try: msg = parseEnum[Message](getMessage())
      except:
        echo "spesiaaali: ", getMessage()
        echo "len: ", ws.requestlen
        return
      case msg:
      of imodd:
        oddclients.add(thesocket)
        reports.atomicInc()
        success = true
      of imeven:
        evenclients.add(thesocket)
        reports.atomicInc()
        success = true
      of oddtoodds: success = wsserver.send(oddclients, $oddtoodds, timeoutsecs = 10)
      of eventoevens: success = wsserver.send(evenclients, $eventoevens, timeoutsecs = 10)
    except:
      echo "Server failed: ", getCurrentExceptionMsg()
      doShutdown()
    if not success and not closing and not shuttingdown:
      echo "System overload"
      doShutdown()


proc clientHandler(client: WebsocketClient) =
  if closing: return
  receivecount.atomicInc()
  if receivecount.load mod 1000 == 0:
    # let msg = getMessage()
    # echo client.socket, " got msg: ", msg
    echo receivecount.load, " messages received"
  if receivecount.load == TotalMessageCount: doShutdown() 


proc threadFunc() {.thread, nimcall, gcsafe.} =
  {.gcsafe.}:
    for i in 1 .. MessageCount:
      if shuttingdown: return
      sleep(1) # lets cheat a little
      let clientid = rand(ClientCount - 1) + 1
      let IAmOdd = clientid mod 2 == 1
      let msg = if IAmOdd: $oddtoodds else: $eventoevens
      if not clientele.clients[clientid].send(msg):
        if not closing:
          echo "System overload"
          doShutdown()
        break
      if IAmOdd: sendcount.atomicInc(oddclients.len)
      else: sendcount.atomicInc(evenclients.len)
      if sendcount.load mod 10000 == 0:
        if not closing: echo sendcount.load, " messages sent"


#let wsServer = newWebSocketServer(nil, nil, serverHandler, nil, TRACE)
let wsServer = newWebSocketServer(nil, nil, serverHandler)
if not wsServer.start(5050): quit()
#clientele = newWebsocketClientele(loglevel = TRACE)
clientele = newWebsocketClientele()
clientele.sockettimeoutms = 10000
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
echo "Aiming to send ", TotalMessageCount, " messages..."  
while reports.load < ClientCount: sleep(10)
for i in 1 .. ThreadCount: createThread(threads[i], threadFunc)
joinThreads(threads)
sleep(3000)
echo sendcount.load, " messages sent"
if sendcount.load < TotalMessageCount:
  echo "Could not send all messages..."
joinThread(wsserver.thread)