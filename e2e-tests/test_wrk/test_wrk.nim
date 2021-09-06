import guildenstern/ctxheader
var server = new GuildenServer
server.initHeaderCtx(proc(ctx: HttpCtx) = ctx.reply(Http204) , 5050, false)
server.serve(1)
