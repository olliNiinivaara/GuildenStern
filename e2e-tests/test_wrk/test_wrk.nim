# nim -b:cpp r --threads:on --gc:arc -d:danger -d:threadsafe test_wrk

import osproc, streams, strutils, guildenstern/[ctxtimer, ctxheader]

var server = new GuildenServer

proc doWrk() =
  {.gcsafe.}: server.removeTimerCtx(doWrk)
  try:
    let wrk = startProcess("wrkbin", "", ["-t16", "-c32",  "-d10s", "--latency",  "http://127.0.0.1:5050"])
    let result = wrk.outputStream().readAll()
    echo result
    if "Socket errors" in result: quit(-1)
    shutdown()
  except: quit(-123)

server.initTimerCtx(1, doWrk)
server.initHeaderCtx(proc(ctx: HttpCtx) = ctx.reply(Http204) , 5050, false)
server.serve(32)