const ClientCount = 10
from os import sleep

#-----------------

import guildenstern/websocketserver
when epollSupported(): import guildenstern/epolldispatcher
else: import guildenstern/dispatcher

proc serverReceive() =
  let message = getMessage()
  echo "server got: ", message
  if message == "close?": wsserver.send(thesocket, "close!")
  else: wsserver.send(thesocket, "Ok!")

let server = newWebsocketServer(receive = serverReceive)
if not server.start(8080): quit()

#-----------------

import guildenstern/websocketclient

proc clientReceive(client: WebsocketClient) =
  let message = getMessage()
  echo "client ", $client.id, " got: ",message
  if message == "close!": shutdown()

let clientele = newWebsocketClientele()

proc run() =
  for i in 1..ClientCount:
    let client = clientele.newWebsocketClient("http://0.0.0.0:8080", clientReceive)
    if not client.connect(): quit()
    client.send("this comes from client " & $client.id)
  sleep(100)
  clientele.clients[1].send("close?")

#-----------------

if clientele.start():
  run()
  joinThread(server.thread)