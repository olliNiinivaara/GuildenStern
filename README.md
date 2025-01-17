# GuildenStern

Modular multithreading HTTP/1.1 + WebSocket upstream server framework for POSIXy OSs (Linux, BSD, MacOS).

## Documentation

https://olliniinivaara.github.io/GuildenStern/theindex.html

## Example 1: hello world

```nim
import guildenstern/[dispatcher, httpserver]
let server = newHttpServer(proc() = reply "hello world")
if server.start(8080): joinThread(server.thread)
```

## Example 2: Partitioning work to fine-tuned servers

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
    let client = clientele.newWebsocketClient("ws://0.0.0.0:8080", clientReceive)
    if not client.connect(): quit()
    client.send("this comes from client " & $client.id)
  sleep(100)
  clientele.clients[1].send("close?")

#-----------------

if clientele.start():
  run()
  joinThread(server.thread)
```

## Release notes, 8.0.0 (2025-01-17)

### breaking changes
- dispatcher's *start* proc now returns *bool* that has to be handled
- *LogCallback* takes also source as parameter (breaking only if you have been using a custom logger procedure)
- socketdata *flags* parameter is not directly accessible anymore, but there are new *getFlags* and *setFlags* procs (only affects those who created new server components)
- new global convenience template *thesocket*, so you don't need write *socketcontext.socket*, *http.socket* or *ws.socket* (breaking only if you were already using a variable named *thesocket*)

### major changes
- new robust *epolldispatcher* available for platforms that support epoll (e.g. Linux)
- new *websocketclient* module available, that let's you test your websocket servers easily (for inspiration, check the new *wsclienttest* and *wsmulticasttest* files in the examples folder)
- SocketData is not anymore available in socketcontext. Instead, *server*, *socket* and *customdata* are directly available in the socketcontext.There is a convenience *socketdata* proc that makes the redirection, so existing code should not break
- *OnCloseSocketCallback* that offers socketdata as parameter is deprecated (but works). Switch to new *OnCloseSocketCallback* that offers server and socket directly as parameters
- new *threadFinalizerCallback* that is triggered for every worker thread just before they stop running
- various stability improvements

### minor changes
- the --d:threadsafe compiler switch is not needed anymore
- all-around better logging
- log messages also include server id and thread id
- servers can work in client mode when port number 0 is used
- new dispatcher proc *registerSocket* for adding sockets to servers working in client mode
- dispatchers close themselves more gracefully, waiting up to 10 seconds for workerthreads to finish
- if dispatcher fails to start, returns false instead of shutting down everything
- various internal improvements for those who write new server components
- new error code EFault for detecting memory corruption (faulty pointer inputs to posix procs)
- new static func *epollSupported* in guildenserver for checking if epoll is supported
- *suspend* proc now needs also the server as parameter. The old suspend exists for backward compatibility, but it always only sleeps
- socket closing is always logged, with suitable log level depending on cause
- *closeOtherSocket* renamed to *closeSocket* (closeOthersocket is deprecated, and just redirects to closeSocket)
- HttpServers do not close the socket, if an empty request is received (because it might be a keep-alive packet)
- reply messages do not need to have and address anymore (constants accepted)
- WebsocketServer supports running in client mode (triggered, when new *clientmaskkey* parameter is set in *initWebsocketServer* proc)
- WebsocketServer now hails the *sockettimeoutms* parameter when receiving messages
- WebsocketServer has new *send* proc for sending a message to many clients simultaneously, that takes *failedsockets: var seq[SocketHandle]* parameter, and returns in it all sockets that failed to receive the message. Consult *serverHandler* proc in wsmulticasttest as an example
