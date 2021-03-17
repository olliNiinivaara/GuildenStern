# This example shows how to implement a custom handler
# The receive and reply procs are written just for demonstration purposes

import guildenstern
from posix import recv,send

type
  MyCustomCtx* = ref object of Ctx
    buf: string
    requestlen: int

var ctx {.threadvar.}: MyCustomCtx

proc receive(): bool  =
  while true:
    if shuttingdown: return false
    let ret = recv(ctx.socketdata.socket, addr ctx.buf[ctx.requestlen], MaxHeaderLength + 1, 0)
    if ret < 1 or ret == MaxHeaderLength + 1: (ctx.closeSocket(NetErrored) ; return false)
    ctx.requestlen += ret
    if ctx.buf[ctx.requestlen-4] == '\c' and ctx.buf[ctx.requestlen-3] == '\l' and
     ctx.buf[ctx.requestlen-2] == '\c' and ctx.buf[ctx.requestlen-1] == '\l': break
  return ctx.requestlen > 0

proc reply() =
  const content = "Hello World!"
  var response = "HTTP/1.1 200 OK\r\LContent-Length: " & $content.len & "\r\L\r\L" & content
  let ret = send(ctx.socketdata.socket, addr response[0], response.len, 0)
  if ret != response.len: ctx.closeSocket(NetErrored)
  
proc handleCustomRequest(gs: ptr GuildenServer, socketdata: ptr SocketData) {.gcsafe, raises: [].} =
  if ctx == nil:
    ctx = new MyCustomCtx
    ctx.buf = newString(MaxRequestLength + 1)
  ctx.socketdata = socketdata
  ctx.requestlen = 0
  if receive(): reply()
    
var server = new GuildenServer
discard server.registerHandler(handleCustomRequest, 8080, "mycustomprotocol")
server.serve()