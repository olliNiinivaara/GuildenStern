# GuildenStern

Modular multithreading HTTP/1.1 + WebSocket upstream server framework

## Example

```nim
# nim r --d:release --d:threadsafe thisexample

import cgi, guildenstern/[dispatcher, httpserver]
     
proc handleGet() =
  echo "method: ", getMethod()
  echo "uri: ", getUri()
  if isUri("/favicon.ico"): reply(Http204)
  else:
    let html = """
      <!doctype html><title>GuildenStern Example</title><body>
      <form action="http://localhost:5051" method="post" accept-charset="utf-8">
      <input name="say" value="Hi"><button>Send"""
    reply(html)

proc handlePost() =
  echo "client said: ", readData(getBody()).getOrDefault("say")
  reply(Http303, ["location: " & http.headers.getOrDefault("origin")])
  
let getserver = newHttpServer(handleGet, contenttype = NoBody)
let postserver = newHttpServer(handlePost, loglevel = INFO, headerfields = ["origin"])
getserver.start(5050)
postserver.start(5051, threadpoolsize = 20, maxactivethreadcount = 10)
joinThreads(getserver.thread, postserver.thread)
```

## Documentation

https://olliniinivaara.github.io/GuildenStern/theindex.html


## Installation

### POSIXy OSs (Linux, BSD, MacOS):
atlas use GuildenStern

### Windows™:

- step 1: Install [WSL](https://learn.microsoft.com/en-us/windows/wsl/install)
- step 2: atlas use GuildenStern


## Release notes, 7.3.0 (2024-12-16)

### Improved websocket close handshaking:
- New proc *sendClose* for sending a close frame to a client with a statuscode
- When a close frame from a client is received and before the socket is closed, an acknowledging close frame is now always automatically replied using the statuscode set by the client
- When websocket is closed by client, the statuscode is now available in *onCloseSocketCallback* via the *msg* parameter
- updated example `websockettest` demonstrating these improvements
