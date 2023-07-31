# nim r --threads:on --gc:orc -d:danger -d:threadsafe test_wrk

#[example result to expect:
Running 10s test @ http://127.0.0.1:5050
  4 threads and 100 connections
  Thread Stats   Avg      Stdev     Max   +/- Stdev
    Latency   228.50us  126.53us   7.43ms   87.58%
    Req/Sec   107.27k     8.56k  198.00k    81.84%
  Latency Distribution
     50%  206.00us
     75%  226.00us
     90%  355.00us
     99%  722.00us
  4289333 requests in 10.10s, 155.44MB read
Requests/sec: 424701.07
Transfer/sec:     15.39MB
]#

import osproc, streams, strutils, guildenstern/[dispatcher, httpserver]

when compileOption("profiler"): import nimprof


proc doWrk(): int =
  try:
    let wrk = startProcess("wrkbin", "", ["-t4", "-c100",  "-d10s", "--latency",  "http://127.0.0.1:5050"])
    let results = wrk.outputStream().readAll()
    echo results
    shutdown()
    if "Socket errors" in results: return -1
  except Exception: return -123
  return 0

proc handle() = reply(Http200)

proc run() =
  let server = newHttpServer(handle, false, false, false)
  server.start(5050, 4)
  let errorcode = doWrk()
  joinThread(server.thread)
  quit(errorcode)

run()