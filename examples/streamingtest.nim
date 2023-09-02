import guildenstern/[dispatcher, streamingserver]

let html = """<!doctype html><title></title><body>
  <form action="/upload" method="post" enctype="multipart/form-data" accept-charset="utf-8">
  <input type="file" id="file" name="file"><input type="submit">"""

proc handleUpload() =
  let ok = "ok"
  if not startReceiveMultipart(giveupSecs = 2): (reply(Http400); return)
  while true:
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
    else: reply(Http200, html)
    
let server = newStreamingServer(onRequest)
server.start(5050)
joinThread(server.thread)