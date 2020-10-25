# nim c -r --gc:arc --d:danger --threads:on --d:threadsafe guildentest.nim


from httpcore import Http200, Http404, Http411

import guildenstern, guildenstern/httphandler


const port = 8080


proc onRequest(h: HttpHandler) =
  if h.isMethod "GET":
    if h.isPath "/plaintext":
      const data = "Hello, World!"
      const headers = "Content-Type: text/plain"
      h.reply(Http200, data, headers)
    else: h.replyCode(Http404)

proc onError(h: Handler) =
  echo h.currentexceptionmsg

proc startServer(multi: bool) {.thread, nimcall, gcsafe.} =
  let server = newGuildenServer()
  initHttpHandling(server, onRequest)
  registerHttpErrorhandler(server, onError)
  serve(server, port, multi)

when ismainmodule:
  startServer(true)
  #var threads: array[4, Thread[void]]
  #for i in 0 .. 3:
  #  createThread(threads[i], startServer)
  #  joinThreads(threads)
  


#[proc onRequest(h: GuildenVars) =
  if h.isMethod "GET":
    if h.isPath "/plaintext":
      const data = "Hello, World!"
      const headers = "Content-Type: text/plain"
      h.reply(Http200, data, headers)
    else: h.replyCode(Http404)

proc startServer(port: int) =
  let server = new GuildenServer
  server.registerHttphandler(onRequest, [])
  serve[GuildenVars](server, port)

proc startServer8080*() = startServer(8080)

when ismainmodule:
  let port = 8080
  echo "GuildenServer listening on port ", port
  startServer(port)]#
