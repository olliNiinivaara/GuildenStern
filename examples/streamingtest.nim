import guildenstern/[dispatcher, streamingserver]

import segfaults

let html = """<!doctype html><title>StreamCtx</title><body>
  <form action="/upload" method="post" enctype="multipart/form-data" accept-charset="utf-8">
  <input type="file" id="file" name="file">
  <input type="submit">"""; let ok = "ok"

proc handleGet() =
  while hasData(): echo receiveChunk()
  reply(Http200, html)

proc handleUpload() =
  if not startReceiveMultipart(giveupSecs = 2): (reply(Http400); return)
  while true:
    suspend(3000)
    let (state , chunk) = receiveMultipart()
    case state
      of Fail: break
      of TryAgain: continue
      of Progress: echo chunk
      of Complete: (reply(Http200, ok); break)
  shutdown()

proc onRequest() =
  {.gcsafe.}:
    if startsUri("/upload"): handleUpload()
    else: handleGet()
    
let server = newStreamingServer(onRequest)
server.start(5050)
joinThread(server.thread)