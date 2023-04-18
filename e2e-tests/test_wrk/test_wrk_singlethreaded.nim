# nim c --gc:orc -d:danger test_wrk_singlethreaded.nim
# ./test_wrk_singlethreaded &
# ./wrkbin -t1 -c10000 -d10s --latency http://127.0.0.1:5050 && kill $!

#[example result:

Running 10s test @ http://127.0.0.1:5050
  1 threads and 10000 connections
  Thread Stats   Avg      Stdev     Max   +/- Stdev
    Latency    69.13ms   33.34ms 903.44ms   96.26%
    Req/Sec   104.81k    10.38k  139.34k    85.19%
  Latency Distribution
     50%   64.16ms
     75%   65.75ms
     90%   76.26ms
     99%  230.91ms
  1029726 requests in 10.12s, 26.59MB read
Requests/sec: 101738.78
Transfer/sec:      2.63MB  
]#

when compileOption("profiler"):
  import nimprof

import guildenstern/ctxheader

var server = new GuildenServer
server.initHeaderCtx(
  proc(ctx: HttpCtx) = ctx.reply(Http204)
  , 5050, false)
server.serve(1)