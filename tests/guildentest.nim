# nim c -r --gc:arc --d:danger --threads:on --d:threadsafe guildentest.nim


from httpcore import Http200, Http404, Http411

import guildenstern #, guildenstern/httphandler
import guildenstern/headerslurper

const
  HeaderSlurping = 1.HandlerType

import nimprof

#[proc onRequest(h: HttpHandler) {.gcsafe.} =
  if h.isMethod "GET":
    if h.isPath "/plaintext":
      let data = "Hello, World!"
      let headers = "Content-Type: text/plain"
      h.reply(Http200, unsafeAddr data, unsafeAddr headers)
    else: h.replyCode(Http404)
    if h.currentexceptionmsg != "": echo "error: " & h.currentexceptionmsg]#

proc onRequest(h: HeaderSlurper) =
  h.replyEmpty()

proc onError(h: Handler) =
  echo h.currentexceptionmsg

proc startServer() {.thread, nimcall, gcsafe.} =
  let server = newGuildenServer([(8080, HeaderSlurping)])
  initHeaderSlurping(HeaderSlurping, server, onRequest)
  registerSlurperErrorhandler(server, onError)
  server.serve(false)

when ismainmodule:
  startServer()
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
