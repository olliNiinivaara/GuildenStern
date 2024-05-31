# <span style="color:red">VERSION 7, In progress</span>

# GuildenStern
Modular multithreading Linux HTTP + WebSocket server

## Example

```nim
# nim r --d:release --d:threadsafe thisexample

import cgi, guildenstern/[dispatcher, httpserver]
     
proc handleGet() =
  echo "method: ", getMethod() 
  echo "uri: ", getUri()
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
postserver.start(5051, threadpoolsize = 200, maxactivethreadcount = 20)
joinThreads(getserver.thread, postserver.thread)
```

## Documentation

https://olliniinivaara.github.io/GuildenStern/index.html

## Installation

atlas use GuildenStern

## Release notes, 7.0.0 (2024-06-??)

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

### Improvements to websocket
- If you do not accept a connection (reply *false* in *upgradeCallback*), a HTTP 400 reply is now automatically sent before the socket is closed
- PONG is now automatically replied to PINGs, *sendPong* proc is removed
- default reply values changed, now *timeoutsecs* = 10, *sleepmillisecs* = 10
- if all in-flight receivers are blocking, now sleeps for (*sleepmillisecs* * in-flight receiver count) milliseconds
- fixed *isMessage* bug

