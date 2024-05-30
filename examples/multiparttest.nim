from std/os import sleep
import guildenstern/[dispatcher, httpserver, multipartserver]

proc interruptUpload(reason: string) =
  echo reason
  reply(Http500)
  sleep(200)
  closeSocket(CloseCalled, reason)

proc startFileUpload(filename: string, file: var File): bool =
  const acceptedFileTypes = @["application/pdf", "text/plain"]
  if http.headers.getOrDefault("content-type") notin acceptedFileTypes:
    interruptUpload(filename & " has wrong content type")
    return false
  if not file.open(filename, fmWrite):
    interruptUpload(filename & " not writeable")
    return false
  return true
  
proc handleUpload() =
  var
    name, msg, subject: string
    file1, file2: File
    file1opened, file2opened: bool
  echo "=============="      
  for (state , chunk) in receiveParts():
    case state:
      of HeaderReady:
        var filename: string
        (name , filename) = parseContentDisposition()
        if filename == "": continue
        case name:
          of "file1":
            if not startFileUpload("file1", file1): return
            file1opened = true
          of "file2":
            if not startFileUpload("file2", file2): return
            file2opened = true       
          else:
            interruptUpload(filename & " is wrong")
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
          of "file1":
            if file1opened: echo "file1 written"
          of "file2":
            if file2opened: echo "file2 written"
          of "msg": echo "msg: ", msg
          of "subject": echo "subject: ", subject
          else: discard
      of Failed:
        echo "failed: ", chunk
        return
      of Completed: discard
  if file1opened: file1.close()
  if file2opened: file2.close()
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