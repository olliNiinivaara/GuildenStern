# GuildenStern

Modular multithreading HTTP/1.1 + WebSocket upstream server framework for POSIXy OSs (Linux, BSD, MacOS).

## Example 1: hello world

```nim
import guildenstern/[dispatcher, httpserver]
let server = newHttpServer(proc() = reply "hello world")
if server.start(8080): joinThread(server.thread)
```

## Example 2: Two different servers running in different ports

```nim
# nim r --d:release --mm:atomicArc thisexample

import cgi, guildenstern/[dispatcher, epolldispatcher, httpserver]
     
proc handleGet() =
  echo "method: ", getMethod()
  echo "uri: ", getUri()
  if isUri("/favicon.ico"): reply(Http204)
  else:
    reply """
      <!doctype html><title>GuildenStern Example</title><body>
      <form action="http://localhost:5051" method="post" accept-charset="utf-8">
      <input name="say" value="Hi"><button>Send"""

proc handlePost() =
  echo "client said: ", readData(getBody()).getOrDefault("say")
  reply(Http303, ["location: " & http.headers.getOrDefault("origin")])
  
let getserver = newHttpServer(handleGet, contenttype = NoBody)
let postserver = newHttpServer(handlePost, loglevel = INFO, headerfields = ["origin"])
if not dispatcher.start(getserver, 5050): quit()
if not epolldispatcher.start(postserver, 5051, threadpoolsize = 20): quit()
joinThreads(getserver.thread, postserver.thread)
```

## Example 3: Websocket server and clients discussing

```nim
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
```

## Documentation

https://olliniinivaara.github.io/GuildenStern/theindex.html


## Release notes, 8.0.0 (2025-01-??)

