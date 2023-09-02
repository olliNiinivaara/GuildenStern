import guildenstern/[dispatcher, streamingserver]
from os import sleep

proc onRequest() =
  var thisisyourdata = "forewords"
  for i in 0 .. 50000: thisisyourdata &= $i
  thisisyourdata &= "finalwords"
  if isUri("/favicon.ico"): reply(Http204)
  elif isUri("/"):
    let html = """<!doctype html><title></title><body>
    <form action="/post" method="POST" accept-charset="utf-8">
    <input type="submit" name="post" value="""" & thisisyourdata & """">"""
    reply(html)
  else:
    var body: string
    var trials = 0
    for (state , chunk) in receiveInChunks():
      case state:
        of TryAgain:
          suspend(100)
          trials += 1
          if trials > 100: (closeSocket() ; break)
        of Fail: reply(Http400)
        of Progress: body &= chunk
        of Complete:
          assert(body.len == "post=".len + thisisyourdata.len)
          echo body[0 .. 20] & "..." & body[(body.len - 20) ..< body.len]
          reply(Http204)
    sleep(100)
    shutdown()

let server = newStreamingServer(onRequest)
server.start(5050)
joinThread(server.thread)