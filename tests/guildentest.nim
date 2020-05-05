#
# weave:
# nim c -r --d:danger --threads:on --d:threadsafe -d:WEAVE_NUM_THREADS=4 guildentest.nim
#
# weave and statistics:
# nim c --d:danger --threads:on --d:threadsafe -d:WEAVE_NUM_THREADS=4 -d:WV_metrics -d:WV_profile guildentest.nim ; ./guildentest
#
# single-threaded:
# nim c -r --d:danger guildentest.nim
#

from httpcore import Http200, Http404

import guildenstern

proc onRequest(c: GuildenVars) =
  if c.isMethod "GET":
    if c.isPath "/plaintext":
      const data = "Hello, World!"
      const headers = "Content-Type: text/plain"
      c.reply(Http200, data, headers)
    else: c.replyCode(Http404)

proc startServer(port: int) =
  let server = new GuildenServer
  server.registerHttphandler(onRequest, [])
  serve[GuildenVars](server, port)

proc startServer8080*() = startServer(8080)

when ismainmodule:
  let port = 8080
  echo "GuildenServer listening on port ", port
  startServer(port)