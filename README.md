![Tests](https://github.com/olliNiinivaara/GuildenStern/workflows/Tests/badge.svg)

# GuildenStern
Modular multithreading Linux HTTP + WebSocket server

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
server.serve(loglevel = DEBUG)
```

## Documentation
[User Guide](http://olliNiinivaara.github.io/GuildenStern/)

[How to implement a custom handler](https://github.com/olliNiinivaara/GuildenStern/blob/master/docs/customhandler.nim)

## Installation

`nimble install guildenstern`

## Features

- Modular architecture means simpler codebase, easy customization and more opportunities for performance optimization
- Every request is served in dedicated thread - scales vertically and requests cannot stall each other 
- Can listen to multiple ports with different handlers

## Release notes, 5.0.0 (2021-09-21)

- Breaking change: If you use timer handlers, you must now import guildenstern/timerctx
- Refactored timer handlers to use same logic as other handlers
- New proc *removeTimerCtx* for stopping timers
- new logging implementation supporting multiple log levels, -d:fulldebug obsoleted
- performance optimization: threadpool now incorporates a (lock-free) task queue