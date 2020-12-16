import guildenstern/ctxheader

var c: int

proc process(ctx: HttpCtx) =
  ctx.reply(Http204)
  if ctx.isUri("/shutdown"):
    echo "shutdown received"
    shutdown()
  c.atomicInc

proc onClose(ctx: Ctx, cause: SocketCloseCause, msg: string){.gcsafe, nimcall, raises: [].} =
  echo c, " ", cause, " ", msg, ": ", ctx.socketdata.socket

var server = new GuildenServer
server.initHeaderCtx(process, 5050)
server.registerConnectionclosedhandler(onClose)
server.serve()
