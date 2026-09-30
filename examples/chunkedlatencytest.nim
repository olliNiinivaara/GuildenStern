## Demonstrates a ~200ms delay before the first byte of every chunked response, caused by
## replyFinishChunked() sending its final (real-data) terminator segment corked (MSG_MORE) instead of
## uncorked - the zero-length send in replyFinish() cannot flush already-corked data, so the whole
## response sits until Linux's own cork timeout releases it.
##
## Before the fix, every request below takes ~200ms end to end (all server-side work completes in
## microseconds - confirmed with timestamps around replyStartChunked/replyContinueChunked - the delay is
## entirely on the wire, between the server's last successful send() call and the client receiving any
## byte of the response). After the fix, requests complete in well under 1ms locally.
##
## Try: for i in 1 2 3 4 5; do curl -s -o /dev/null -w "%{time_total}\n" http://localhost:5061/hello; done
import guildenstern/[dispatcher, httpserver]

proc onRequest() =
  if not replyStartChunked(): (shutdown(); return)
  discard replyContinueChunked("hello")
  replyFinishChunked()

let s = newHttpServer(onRequest)
s.start(5061)
joinThread(s.thread)
