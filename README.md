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

### 2.0.0 release notes
- CloseCallback has new parameter `socket` that allows receiving closecallbacks also for sockets other than current request
- new `proc closeOtherSocket` allows closing sockets other than current request (such as websockets)
- new `proc getProtocolName` for checking type (http, websocket, ...) of closed sockets other than current request
- new `proc multiSendWs` uses non-blocking I/O to cleverly deliver multiple messages to multiple websockets in one shot (even when some receivers have sloppy connections)
- avoids weird behavior by refusing to use socket address or port that is already in use
- unreserves sockets at shutdown with SO_LINGER 0 close option so that they can be reused immediately
- `posix.SocketHandle` is used instead of `nativesockets.SocketHandle`
- `posix.INVALID_SOCKET` is used instead of `nativesockets.osInvalidSocket`


## Baseline latency ([Intel i5-760](https://ark.intel.com/content/www/us/en/ark/products/48496/intel-core-i5-760-processor-8m-cache-2-80-ghz.html))

```
olli@nexus:~/Nim/Testit/SuberGuilden$ wrk -d10 -t2 --latency http://localhost:5050
Running 10s test @ http://localhost:5050
  2 threads and 10 connections
  Thread Stats   Avg      Stdev     Max   +/- Stdev
    Latency   146.83us   40.33us   4.28ms   98.10%
    Req/Sec    33.88k   745.53    34.91k    92.57%
  Latency Distribution
     50%  144.00us
     75%  147.00us
     90%  149.00us
     99%  199.00us
  681017 requests in 10.10s, 536.46MB read
```