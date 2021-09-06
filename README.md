![Tests](https://github.com/olliNiinivaara/GuildenStern/workflows/Tests/badge.svg)

# GuildenStern
Modular multithreading Linux HTTP server

## Example

```nim
# nim c -r --gc:arc --d:release --threads:on --d:threadsafe example.nim

import cgi, strtabs, httpcore, guildenstern/[ctxheader, ctxbody]
     
proc handleGet(ctx: HttpCtx) =
  let html = """
    <!doctype html><title>GuildenStern Example</title><body>
    <form action="http://localhost:5051" method="post">
    <input name="say" id="say" value="Hi"><button>Send"""
  ctx.reply(html)

proc handlePost(ctx: HttpCtx) =
  try: echo readData(ctx.getBody()).getOrDefault("say")
  except: (ctx.reply(Http400) ; return)
  ctx.reply(HttpCode(303), ["location: http://localhost:5050"])

var server = new GuildenServer
server.initHeaderCtx(handleGet, 5050, false)
server.initBodyCtx(handlePost, 5051)
echo "GuildenStern HTTP server serving at 5050"
server.serve()
```

## Documentation
[User Guide](http://olliNiinivaara.github.io/GuildenStern/)

[How to implement a custom handler](https://github.com/olliNiinivaara/GuildenStern/blob/master/docs/customhandler.nim)

## Installation

`nimble install guildenstern`

## Features

- Modular architecture means simpler codebase, easy customization and more opportunities for performance optimization
- Every request is served in dedicated thread, requests won't slow each other down 
- Preemptive multithreading guarantees low latencies by fair access to CPU cores
- Can listen to multiple ports with different handlers
- Supports streaming requests, streaming replies, and websocket
- Supports --gc:arc, doesn't need asyncdispatch
- Runs in single-threaded mode, too

## Release notes, 4.0.0 (2021-09-06)

- *Serve* proc now accepts the amount of worker threads to use 
- *registerThreadInitializer* is now a global proc (not tied to a GuildenServer)
- *ctxWs web socket upgrade* now accepts a callback closure (instead of initial message to send)
- Can listen to multiple ports even with same handler (just register different handler callback proc for each port)
- new *ctxBody* handler for efficiently handling POST requests.
- new *isRequest* proc for efficiently inspecting request content
- new custom threadpool
- code fixes and cleanups
