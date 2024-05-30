# nim r -d:danger -d:threadsafe wrktest

#[example result to expect:
  4 threads and 100 connections
  Thread Stats   Avg      Stdev     Max   +/- Stdev
    Latency   186.95us  103.47us   4.30ms   95.75%
    Req/Sec   128.91k    11.69k  195.25k    67.66%
  Latency Distribution
     50%  171.00us
     75%  183.00us
     90%  191.00us
     99%  750.00us
  5152524 requests in 10.10s, 132.67MB read
Requests/sec: 510173.37
Transfer/sec:     13.14MB
]#

import osproc, streams, strutils, guildenstern/[dispatcher, httpserver]

proc doWrk(): int =
  try:
    let wrk = startProcess("wrkbin", "", ["-t4", "-c100",  "-d10s", "--latency", "http://127.0.0.1:5050"])
    let results = wrk.outputStream().readAll()
    echo results
    shutdown()
    if "Socket errors" in results: return -1
  except Exception: return -123
  return 0

proc handle() = reply(Http204)

proc run() =
  let httpserver = newHttpServer(handle, NONE, false, NoBody)
  httpserver.start(5050)
  let errorcode = doWrk()
  joinThread(httpserver.thread)
  quit(errorcode)

run()