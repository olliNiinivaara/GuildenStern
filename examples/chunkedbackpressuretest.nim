## Reproduces a data-corruption bug in replyContinueChunked() under backpressure (slow reader / EAGAIN).
##
## `delivered += len` in replyContinueChunked (httpresponse.nim) adds tryWriteToSocket's raw send()
## return value to `delivered` unconditionally - including when that value is a negative errno-shaped
## sentinel (state == TryAgain or Fail, not a byte count). Every EAGAIN retry then corrupts the chunk's
## delivery offset. This is invisible on a fast/local reader (each chunk is usually sent in one call,
## state == Complete immediately) - it needs a reader slow enough to make the server's non-blocking
## writes actually hit EAGAIN a few times per chunk.
##
## To reproduce: run this server, then fetch it with a rate limit low enough to force retries, e.g.:
##   curl --limit-rate 2M http://localhost:5060/big -o /dev/null
## Before the fix: curl fails with "Malformed encoding found in chunked-encoding" partway through.
## After the fix: the full payload downloads correctly.
import std/strutils
import guildenstern/[dispatcher, httpserver]

proc onRequest() =
  if isUri("/favicon.ico"): (reply(Http204); return)
  if not replyStartChunked(): (shutdown(); return)
  let block64k = 'x'.repeat(65536)
  for _ in 0 ..< 2048:    # 128 MiB total: enough for a rate-limited reader to induce many EAGAIN retries
    if not replyContinueChunked(block64k): return
  replyFinishChunked()

let s = newHttpServer(onRequest)
s.start(5060)
joinThread(s.thread)
