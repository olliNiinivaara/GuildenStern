# nim c -r --d:danger --threads:on --d:threadsafe wstest.nim
# use k6script.js as test client

import guildenstern

proc handleHttp(gv: GuildenVars) =
  gv.upgradeHttpToWs()

proc handleWs(gv: GuildenVars) =
  discard gv.write("PONG!")
  discard gv.sendToWs()

proc startServer(port: int) =
  let server = new GuildenServer
  server.registerHttphandler(handleHttp, [])
  server.registerWshandler(handleWs)
  serve[GuildenVars](server, port)

startServer(8080)