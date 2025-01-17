# nim r -mm:atomicArc -d:release wsmulticasttest 

from strutils import parseEnum
from os import sleep
import atomics, random, locks
import guildenstern/[epolldispatcher, websocketserver, websocketclient]

type Message = enum
  imodd = "I am odd"
  imeven = "I am even"
  oddtoodds = "hello all odds"
  eventoevens = "hello all evens"

const
  ClientCount = 50
  MessageCount = 5000

doAssert(ClientCount mod 2 == 0)

var Aim =  MessageCount * ClientCount div 2

var
  oddclients = newSeq[SocketHandle]()
  evenclients = newSeq[SocketHandle]()
  clientlock: Lock
  reports, sendcount, receivecount: Atomic[int]
  closing: bool
  clientele: WebsocketClientele

clientlock.initLock

proc doShutdown() =
  {.gcsafe.}:
    if closing: return
    closing = true
    echo "Shutting down now..."
    sleep(2000)
    for client in clientele.connectedClients():
      client.close(false)
      if client.id mod 10 == 0: echo client.id, "/", ClientCount, " clients closed"
    echo "All connections closed"
    sleep(1000)
    shutdown()
    

var failedsockets {.threadvar.}: seq[SocketHandle]

proc serverHandler() =
  {.gcsafe.}:
    if closing: return
    var msg: Message
    try:
      try: msg = parseEnum[Message](getMessage())
      except:
        echo "weird: ", getMessage()
        return
      case msg:
      of imodd:
        oddclients.add(thesocket)
        reports.atomicInc()
      of imeven:
        evenclients.add(thesocket)
        reports.atomicInc()
      of oddtoodds:
        wsserver.send(oddclients, $oddtoodds, failedsockets)
        sendcount.atomicInc(oddclients.len - failedsockets.len)
      of eventoevens:
        wsserver.send(evenclients, $eventoevens, failedsockets)
        sendcount.atomicInc(evenclients.len - failedsockets.len)
      Aim.atomicDec(failedsockets.len)
      if msg notin [imodd, imeven]:
        if sendcount.load mod 10000 == 0:
          echo sendcount.load, " messages sent"
    except:
      echo "Server failed: ", getCurrentExceptionMsg()
      doShutdown()
    if failedsockets.len > 0:
      echo failedsockets.len, " sockets failed"
      for socket in failedsockets:
        echo "failed socket: ", socket
        clientele.findClient(socket).close(false)
        withLock(clientlock):
          try:
            if msg == oddtoodds: oddclients.delete(oddclients.find(socket))
            else: evenclients.delete(evenclients.find(socket))
          except: discard # already deleted?
      failedsockets.setLen(0)


proc clientHandler(client: WebsocketClient) =
  receivecount.atomicInc()
  if receivecount.load mod 10000 == 0:
    echo receivecount.load, " messages received"
  if receivecount.load >= Aim:
    echo "Done: ", receivecount.load
    doShutdown() 
  

proc sendStarters() =
  {.gcsafe.}:
    for i in 1 .. MessageCount:
      if shuttingdown: return
      sleep(1) # let's cheat a little, otherwise we are DDoS:ing the server
      withlock(clientlock):
        let clientid = rand(ClientCount - 1) + 1
        let client = clientele.clients[clientid]
        let IAmOdd = clientid mod 2 == 1
        let msg = if IAmOdd: $oddtoodds else: $eventoevens
        if not client.isConnected() or not client.send(msg):
          if not closing and client.isConnected():
            echo "System overload for socket ", client.socket
          if IAmOdd: Aim.atomicDec(oddclients.len)
          else: Aim.atomicDec(evenclients.len)    


let wsServer = newWebSocketServer(receive = serverHandler)
if not wsServer.start(5050): quit()
clientele = newWebsocketClientele(bufferlength = 20)
if not clientele.start(): quit()
for i in 1 .. ClientCount:
  let client = clientele.newWebsocketClient("ws://127.0.0.1:5050", clientHandler)
  if not client.connect(): quit("could not connect to server")
  let isodd = i mod 2 == 1
  let msg = if isodd: $imodd else: $imeven
  if not client.send(msg):
    echo "could not report oddity to server"
    doShutdown()
  if i mod 100 == 0: echo i, "/", ClientCount, " clients connected"  
while reports.load < ClientCount: sleep(10)
echo "Aiming to ", Aim
sendStarters()
joinThread(wsserver.thread)