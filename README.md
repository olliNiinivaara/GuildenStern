![Tests](https://github.com/olliNiinivaara/GuildenStern/workflows/Tests/badge.svg)

# GuildenStern
Modular multithreading Linux HTTP server

## Example

```nim
# nim c -r --gc:arc --d:release --threads:on --d:threadsafe example.nim

import guildenstern/ctxfull

let replystring = "hello"

proc handleGet(ctx: HttpCtx, headers: StringTableRef) =
  echo "uri: ", ctx.getUri()
  echo "headers: ", headers
  ctx.reply(Http200, replystring)

var server = new GuildenServer
server.initFullCtx(handleGet, 8080)
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

## Release notes, 4.0.0-rc.1 (2021-08-11)
 
- *registerThreadInitializer* is now a global proc (not tied to a GuildenServer)
- new *setWorkerThreadCount* proc for (optionally) setting the amount of worker threads
- new *isRequest* proc for efficiently inspecting request content
- removed dependency on stdlib threadpool
- code fixes and cleanups
