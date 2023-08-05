import guildenstern/[dispatcher, streamingserver]

proc handleGet() =
  echo "hhh"
  let html = """<!doctype html><title>StreamCtx</title><body>
  <form action="/upload" method="post" enctype="multipart/form-data" accept-charset="utf-8">
  <input type="file" id="file" name="file">
  <input type="submit">"""
  while hasData():
    echo receiveChunk()
  reply(Http200, html)


proc handleUpload() =
  let ok = "ok"
  let failure = "failure"
  if not startReceiveMultipart():
    reply(Http200, failure)
    return
  while true:
    let (state , chunk) = receiveMultipart()
    case state
      of Fail:
        reply(Http200, failure)
        return
      of TryAgain:
        continue
      of Progress:
        echo chunk
      of Complete:
        reply(Http200, ok)
        shutdown()
        break


proc onRequest() =
  if startsUri("/upload"): handleUpload()
  else: handleGet()
    

var server = newStreamingServer(onRequest)
server.start(5050)
echo "Point your browser to localhost:5050"
joinThread(server.thread)
echo "done"