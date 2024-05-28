import guildenstern/[dispatcher, httpserver, multipartserver]


proc handleUpload() =             
  for (state , chunk) in receiveParts():
    var currentpart: string
    case state:
      of HeaderReady:
        echo "header received: ", chunk
        echo "---------------------"
      of PartChunk:
        currentpart &= chunk
      of PartReady:
        echo "TODO"
        echo currentpart
        echo "======================="
        currentpart = ""
      else: discard
  reply(Http204)


proc onRequest() =
  let html = """<!doctype html><title>StreamCtx</title><body>
  <form action="http://localhost:5051/upload" method="post" enctype="multipart/form-data" accept-charset="utf-8">
  <label for="file1">File:</label>
  <input type="file" id="file1" name="file1"><br>
  <label for="subject">Subject:</label>
  <input type="text" id="subject" name="subject"><br>
  <label for="msg">Msg:</label>
  <input type="text" id="msg" name="msg"><br>
  <label for="file2">File:</label>
  <input type="file" id="file2" name="file2"><br>
  <input type="submit">"""
  reply(Http200, html)


let getserver = newHttpServer(onRequest, NOTICE, false, NoBody)
getserver.start(5050)
let uploadserver = newMultipartServer(handleUpload)
uploadserver.start(5051)
joinThreads(getserver.thread, uploadserver.thread)