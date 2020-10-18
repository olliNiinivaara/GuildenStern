#
# threadpool:
# nim c -r --gc:arc --d:danger --threads:on --d:threadsafe guildentest.nim
#
# single-threaded:
# nim c -r --gc:arc --d:danger guildentest.nim
#

from httpcore import Http200, Http404

import guildenstern

proc onRequest(gv: GuildenVars) =
  if gv.isMethod "GET":
    if gv.isPath "/plaintext":
      const data = "Hello, World!"
      const headers = "Content-Type: text/plain"
      gv.reply(Http200, data, headers)
    else: gv.replyCode(Http404)

proc startServer(port: int) =
  let server = new GuildenServer
  server.registerHttphandler(onRequest, [])
  serve[GuildenVars](server, port)

proc startServer8080*() = startServer(8080)

when ismainmodule:
  let port = 8080
  echo "GuildenServer listening on port ", port
  startServer(port)
