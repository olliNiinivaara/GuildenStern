# nim r --threads:on --gc:orc -d:danger -d:threadsafe test_wrk

#[example result:

Running 10s test @ http://127.0.0.1:5050
  8 threads and 5000 connections
  Thread Stats   Avg      Stdev     Max   +/- Stdev
    Latency    45.89ms    4.78ms 149.74ms   93.38%
    Req/Sec    13.59k     0.86k   16.45k    76.01%
  Latency Distribution
     50%   44.45ms
     75%   47.71ms
     90%   48.92ms
     99%   57.32ms
  1071199 requests in 10.04s, 27.69MB read
Requests/sec: 106692.14
Transfer/sec:      2.76MB  
]#

import osproc, streams, strutils, guildenstern/[ctxtimer, ctxheader]

var server = new GuildenServer

proc doWrk() {.raises: [].} =
  try:
    {.gcsafe.}: server.removeTimerCtx(doWrk)
    let wrk = startProcess("wrkbin", "", ["-t8", "-c5000",  "-d10s", "--latency",  "http://127.0.0.1:5050"])
    let result = wrk.outputStream().readAll()
    echo result
    if "Socket errors" in result: quit(-1)
    shutdown()
  except Exception: quit(-123)

server.initTimerCtx(1, doWrk)
server.initHeaderCtx(proc(ctx: HttpCtx) = ctx.reply(Http204) , 5050, false)
server.serve(8)