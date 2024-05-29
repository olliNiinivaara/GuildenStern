from std/strutils import find
from std/os import sleep
import guildenstern/[dispatcher, httpserver, multipartserver]

const acceptedFileTypes = @["application/pdf", "text/plain"]

proc interruptUpload(reason: string) =
  echo reason
  reply(Http500)
  sleep(200)
  closeSocket(CloseCalled, reason)
  
proc handleUpload() =
  var
    name, msg, subject: string
    file1, file2: File
  if not file1.open("file1", fmWrite):
    interruptUpload("file1 not writeable")
    return
  if not file2.open("file2", fmWrite):
    interruptUpload("file2 not writeable")
    return
  defer:
    file1.close()
    file2.close()
  echo "=============="      
  for (state , chunk) in receiveParts():
    case state:
      of HeaderReady:
        let value = http.headers.getOrDefault("content-disposition")
        name = value[17 ..< value.find('"', 18)]
        if name in ["file1", "file2"] and http.headers.getOrDefault("content-type") notin acceptedFileTypes:
          interruptUpload(name & " has wrong content type")
          return
      of BodyChunk:
        try:
          case name:
            of "file1": file1.write(chunk)
            of "file2": file2.write(chunk)
            of "msg": msg &= chunk
            of "subject": subject &= chunk
            else: discard
        except:
          interruptUpload(name & " IO error")
          return
      of BodyReady:
        case name:
          of "file1": echo "file1 written"
          of "file2": echo "file2 written"
          of "msg": echo "msg: ", msg
          of "subject": echo "subject: ", subject
          else: discard
      of Failed:
        echo "failed: ", chunk
        return
      of Completed: discard
  reply(Http303, ["location: http://localhost:5050"])

proc onRequest() =
  let html = """<!doctype html><title>multiparttest</title><body>
  <form action="http://localhost:5051/upload" method="post" enctype="multipart/form-data" accept-charset="utf-8">
  <label for="file1">File:</label>
  <input type="file" id="file1" name='file1'><br>
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