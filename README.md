# GuildenStern

Modular multithreading HTTP/1.1 + WebSocket server framework

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

### 7.x.x series:
https://olliniinivaara.github.io/GuildenStern/7/theindex.html

### 6.1.0:
https://olliniinivaara.github.io/GuildenStern/6.1.0/index.html



## Installation

### POSIXy OSs (Linux, BSD, MacOS):
atlas use GuildenStern

### Windows™:

- step 1: Install [WSL](https://learn.microsoft.com/en-us/windows/wsl/install)
- step 2: atlas use GuildenStern


## Release notes, 7.1.0 (2024-06-10)

- All threads are initialized when their server is started (instead of when they are first used), making the system more [fail-fast](https://en.wikipedia.org/wiki/Fail-fast_system)
- \- is an accepted char in header field names
- WebSocketServer code clean-up: WebSocketContext removed; opcode not anymore visible to end users
- Minor fixes and improvements

## Release notes, 7.0.0 (2024-06-01)

### StreamingServer has been merged to HttpServer
- *receiveInChunks* iterator renamed to *receiveStream*
- *maxrequestlength* renamed to *bufferlength*
- newHttpServer does not anymore take *hascontent* parameter, but instead:
- newHttpServer takes *contenttype* parameter:
  - *Compact* (the default mode): Equivalent to previous hasContent=true mode, where whole request body must fit into the buffer
  - *NoBody*: Equivalent to previous hasContent=false mode, for optimized handling of requests like GET that do not have a body
  - *Streaming*: Equivalent to previous StreamingServer, where you have to read the body yourself using the *receiveStream* iterator 
- *startDownload-continueDownload-finishDownload* combo renamed to *replyStartChunked-replyContinueChunked-replyFinishChunked*

### Header processing streamlined
- *parseHeaders* and *parseAllHeaders* obsoleted
- newHttpServer takes *headerfields* parameter, listing header fields that you need
- in request handlers *http.headers* StringTableRef is automatically populated with needed header key-value pairs

### New MultipartServer
- handles *multipart/form-data* for you as neatly parsed
- operates in streaming manner, allowing huge uploads without blowing up RAM

### Improvements to websockets
- If you do not accept a connection (reply *false* in *upgradeCallback*), a HTTP 400 reply is now automatically sent before the socket is closed
- PONG is now automatically replied to PINGs, *sendPong* proc is removed
- default reply values changed, now *timeoutsecs* = 10, *sleepmillisecs* = 10
- if all in-flight receivers are blocking, now suspends for (*sleepmillisecs* * in-flight receiver count) milliseconds
- fixed *isMessage* bug