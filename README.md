# GuildenStern

Modular multithreading HTTP/1.1 + WebSocket upstream server framework

## Example

```nim
# nim r --d:release --mm:atomicArc thisexample

import cgi, guildenstern/[dispatcher, osdispatcher, httpserver]
     
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
if not osdispatcher.start(postserver, 5051, threadpoolsize = 20): quit()
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


## Release notes, 8.0.0 (2025-01-??)

todo

regression: status code replies are wrong