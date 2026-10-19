# Regression test: a request must not be judged by what an earlier request left in the thread's buffer.
#
# The receive buffer of a thread is reused from request to request. `getContentLength` used to look for "content-length: " in all of it,
# so after a POST had been read, its header stayed behind the end of the next, shorter request (a GET without a body), was found there,
# could not be parsed, and the connection was closed with "could not parse content-length".
#
# Run: nim c -r examples/staleheadertest.nim   (exits with 0 when every request is answered, with 1 otherwise)

import std/[net, strutils, os]
import guildenstern/[dispatcher, httpserver]

const port = 8091

proc handle() =
  reply($getBodylen())          # the length of the body the server received

proc request(raw: string; bodylen: int): bool =
  ## true if the server answers with a 200 and says it received a body of this length
  var s = newSocket()
  defer: s.close()
  try:
    s.connect("127.0.0.1", Port(port), 2000)
    s.send(raw)
    var answer = ""
    while not answer.contains("\r\n\r\n"):
      let part = s.recv(1, 2000)
      if part.len == 0: return false
      answer.add part
    let headEnd = answer.find("\r\n\r\n") + 4
    let lower = answer.toLowerAscii
    let at = lower.find("content-length: ")
    if at < 0: return false
    let want = parseInt(lower[at + 16 ..< lower.find("\r\n", at)])
    while answer.len - headEnd < want:
      let part = s.recv(1, 2000)
      if part.len == 0: return false
      answer.add part
    return answer.startsWith("HTTP/1.1 200") and answer[headEnd .. ^1] == $bodylen
  except CatchableError:
    return false

let server = newHttpServer(handle)    # the default: Compact, no headerfields
server.start(port, 2)
sleep 200

var failed = 0
for i in 1 .. 50:
  let body = "x".repeat(20 + i)
  # a header block long enough that "Content-Length" ends up behind the end of a short GET that comes after it
  let post = "POST /a/rather/long/path/to/make/the/request/longer/" & $i & " HTTP/1.1\r\nHost: localhost\r\nUser-Agent: staleheadertest\r\nAccept: */*\r\n" &
             "Content-Type: text/plain\r\nContent-Length: " & $body.len & "\r\nConnection: close\r\n\r\n" & body
  if not request(post, body.len): inc failed
for i in 1 .. 200:
  if not request("GET /h HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n", 0): inc failed

shutdown()
if failed > 0:
  echo "FAILED: ", failed, " of 250 requests were not answered"
  quit 1
echo "ok: all 250 requests were answered"
