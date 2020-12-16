# nim c --d:danger --gc:arc --threads:on --d:threadsafe guildenfibo.nim
# ./guildenfibo &
# ./wrkbin -t8 -c8 -d10s --latency http://127.0.0.1:5050
# kill -INT $!

import random, guildenstern/ctxheader

proc fibonacci(n: int): int =
  result = if n <= 2: 1 else: fibonacci(n - 1) + fibonacci(n - 2)

proc process(ctx: HttpCtx) =
  echo fibonacci(20 + rand(20))
  ctx.reply(Http204)

var server = new GuildenServer
server.initHeaderCtx(process, 5050)
server.serve()
