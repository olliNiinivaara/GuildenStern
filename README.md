# GuildenStern
Modular multithreading Linux HTTP + WebSocket server

## Example

```nim
# nim r --d:release --d:threadsafe thisexample

import cgi, strtabs, guildenstern/[dispatcher, httpserver]
     
proc handleGet() =
  let html = """
    <!doctype html><title>GuildenStern Example</title><body>
    <form action="http://localhost:5051" method="post" accept-charset="utf-8">
    <input name="say" id="say" value="Hi"><button>Send"""
  reply(html)

proc handlePost() =
  try: echo readData(getBody()).getOrDefault("say")
  except: (reply(Http400) ; return)
  reply(Http303, ["location: http://localhost:5050"])

let getserver = newHttpServer(handleGet)
getserver.start(5050)
let postserver = newHttpServer(handlePost)
postserver.start(5051)
joinThreads(getserver.thread, postserver.thread)
```

## Documentation

https://olliniinivaara.github.io/GuildenStern/index.html

[migration guide](https://github.com/olliNiinivaara/GuildenStern/blob/master/docs/migration.md)

## Installation

atlas use GuildenStern


## Release notes, 6.0.0 (2023-08-23)

- major rewrite, breaking changes here and there, consult migration guide and code examples.
- dispatcher(s) can now be replaced just by changing import(s)
- every TCP port is now served by different server, allowing port-by-port configuration of resource usage
- non-blocking I/O with cooperative multithreading now used everywhere
- new suspend procedure for allowing other threads to run also when waiting for I/O in user code
- overall compatibility with Nim version 2.0
- single-threaded mode is no more
- TimerCtx is no more
