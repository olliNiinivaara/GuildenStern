# nimble install httpbeast
# nim c --gc:arc --d:danger --threads:on --d:threadsafe beastfibo.nim
# ./beastfibo &
# ./wrkbin -t8 -c8 -d10s --latency http://127.0.0.1:8080
# kill -INT $!

import random, asyncdispatch, httpbeast

proc fibonacci(n: int): int =
  result = if n <= 2: 1 else: fibonacci(n - 1) + fibonacci(n - 2)

proc onRequest(req: Request): Future[void] =
  echo fibonacci(20 + rand(20))
  req.send(Http204)

run(onRequest)