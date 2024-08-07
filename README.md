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


## Release notes, 7.2.0 (2024-08-07)

- Compatibility with ARM architecture
- Bug fix: getUri etc. in the new compact http server mode
- Bug fix: failed read in the new compact http server mode