# GuildenStern

Modular multithreading HTTP/1.1 + WebSocket upstream server ecosystem for POSIXy OSs (Linux, BSD, MacOS).
Allows concurrently running multiple servers, making your application more efficient, configurable, and fault-tolerant than sticking to a single web server.
Splitting work to many servers supports the [modular monolith](https://arxiv.org/pdf/2401.11867) architecture style.
Out-of-the-box includes two dispatcher implementations and following servers: http, multi-part/form-data, websocket, and websocket client, where the http server has optimized configurations for receiving bodyless, compact and streaming data and for sending in normal and chunked mode.

## Documentation

https://olliniinivaara.github.io/GuildenStern/theindex.html

## Example 1: hello world

```nim
import guildenstern/[dispatcher, httpserver]
let server = newHttpServer(proc() = reply "hello world")
server.start(8080)
joinThread(server.thread)
```

## Example 2: Partitioning work to fine-tuned servers

```nim
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
  
let getserver = newHttpServer(handleGet, loglevel = lvlInfo, contenttype = NoBody)
let postserver = newHttpServer(handlePost, headerfields = ["origin"])
dispatcher.start(getserver, 5050)
epolldispatcher.start(postserver, 5051, threadpoolsize = 20)
joinThreads(getserver.thread, postserver.thread)
```

## Example 3: Websocket server and multiple clients discussing

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
server.start(8080)

#-----------------

import guildenstern/websocketclient

proc clientReceive(client: WebsocketClient) =
  let message = getMessage()
  echo "client ", $client.id, " got: ",message
  if message == "close!": shutdown()

let clientele = newWebsocketClientele()

proc run() =
  for i in 1..ClientCount:
    let client = clientele.newWebsocketClient("ws://0.0.0.0:8080", clientReceive)
    if not client.connect(): quit()
    client.send("this comes from client " & $client.id)
  sleep(100)
  clientele.clients[1].send("close?")

#-----------------

clientele.start()
run()
joinThread(server.thread)
```

## Release notes, 9.0.1 (2026-10-04)
- HTTP chunked response bug fix: if a receiver is slow (causing EAGAIN), do not send corrupted data, but repl(a)y as required
- HTTP chunked response performance fix: send final terminator without MSG_MORE flag
- networking fix in dispatchers: set the NODELAY flag to socket connections


## Release notes, 9.0.0 (2026-07-26)

### breaking changes
- instead of custom LogLevel type, the Level type from std/logging module is used, to achieve compatibility with other logging solutions (such as [Bigsister](https://codeberg.org/olliNiinivaara/Bigsister)). Migrate your code by adding the string "lvl" to all log levels. TRACE becomes lvlTRACE, DEBUG becomes lvlDEBUG, INFO becomes lvlINFO, and so on.
- websocketserver has new initWebsocketClientServer proc for initializing a server in client mode. Previously client mode was inferred from initWebsocketServer proc containing a maskkey argument. That parameter has been removed, and initWebsocketClientServer generates the random mask key automatically

### other
- starting a server (dispatchers and websocketclientele) always returns true, and all error handling is delegated to throwing an exception. The boolean return value is discardable and deprecated
- epolldispatcher robustness improvement, especially concerning the shutdown choreography
- removal of old deprecated features
