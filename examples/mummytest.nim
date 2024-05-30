# nim r --mm:orc --threads:on -d:release -d:threadsafe mummytest

#[example result to expect:
  10 threads and 100 connections
  Thread Stats   Avg      Stdev     Max   +/- Stdev
    Latency    92.39ms   61.38ms 518.03ms   82.64%
    Req/Sec   115.03     26.71   292.00     76.63%
  11441 requests in 10.01s, 726.24KB read
Requests/sec:   1142.83
Transfer/sec:     72.54KB
]#

import osproc, streams, strutils, os, guildenstern/[dispatcher, httpserver]

proc doWrk(): int =
  try:
    let wrk = startProcess("wrkbin", "", ["-t10", "-c100",  "-d10s", "http://127.0.0.1:8080"])
    let results = wrk.outputStream().readAll()
    echo results
    shutdown()
    if "Socket errors" in results: return -1
  except Exception: return -123
  return 0

proc handler() =
  let responseBody = "abcdefghijklmnopqrstuvwxyz"
  if isUri("/"):
    if isMethod("GET"):
      sleep(10)
      reply(Http200, responseBody)
    else:
      reply(Http405)
  else:
    reply(Http404)

proc run() =
  let httpserver = newHttpServer(handler, NONE, true, NoBody)
  httpserver.start(8080, threadpoolsize = 100)
  let errorcode = doWrk()
  joinThread(httpserver.thread)
  quit(errorcode)

run()