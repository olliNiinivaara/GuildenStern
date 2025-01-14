import guildenstern/[dispatcher, httpserver]
from os import sleep

var thisisyourdata: string

proc onGet() =
  {.gcsafe.}:
    reply """<!doctype html><title></title><body>
    <form action="http://localhost:5051" method="POST" accept-charset="utf-8">
    <input type="submit" name="post" value="""" & thisisyourdata & """">"""

proc onPost() =
  var body: string
  var trials = 0
  for (state , chunk) in receiveStream():
    case state:
      of TryAgain:
        server.suspend(100)
        trials += 1
        if trials > 100: (closeSocket() ; break)
      of Fail: discard
      of Progress:
        body &= chunk
        trials = 0
      of Complete:
        {.gcsafe.}: doAssert(body.len == "post=".len + thisisyourdata.len)
        echo body[0 .. 20] & "..." & body[(body.len - 20) ..< body.len]
        reply(Http303, ["location: http://localhost:5050"])
  sleep(100)

thisisyourdata = "forewords"
for i in 0 .. 50000: thisisyourdata &= $i
thisisyourdata &= "finalwords"
let getserver = newHttpServer(onGet, contenttype = NoBody)
let postserver = newHttpServer(onPost, contenttype = Streaming)
if not getserver.start(5050): quit() 
if not postserver.start(5051): quit()
joinThreads(getserver.thread, postserver.thread)