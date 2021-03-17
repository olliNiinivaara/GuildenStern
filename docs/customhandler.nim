# This example shows how to implement a custom handler
# The receive and reply procs are written just for demonstration purposes

import guildenstern
from posix import recv,send

type
  MyCustomCtx* = ref object of Ctx
    buf: string
    requestlen: int

var server = new GuildenServer
var ctx {.threadvar.}: MyCustomCtx

proc receive(): bool  =
  while true:
    if shuttingdown: return false
    let ret = recv(ctx.socketdata.socket, addr ctx.buf[ctx.requestlen], 1 + MaxHeaderLength - ctx.requestlen, 0)
    ctx.requestlen += ret
    if ret < 5 or ctx.requestlen > MaxHeaderLength: (ctx.closeSocket(ProtocolViolated) ; return false)
    if ctx.buf[ctx.requestlen - 4 ..< ctx.requestlen] == "\c\L\c\L": break
  return ctx.requestlen > 0

proc reply() =
  var response = "HTTP/1.1 200 OK\r\LContent-Length: 12\r\L\r\LHello World!"
  let ret = send(ctx.socketdata.socket, addr response[0], response.len, 0)
  if ret != response.len: ctx.closeSocket(ProtocolViolated)

proc initThread() =
  ctx = new MyCustomCtx
  {.gcsafe.}: ctx.gs = addr server
  ctx.buf = newString(MaxHeaderLength + 1)
  
proc handleCustomRequest(gs: ptr GuildenServer, socketdata: ptr SocketData) =
  ctx.socketdata = socketdata
  ctx.requestlen = 0
  if receive(): reply()
    
discard server.registerHandler(handleCustomRequest, 8080, "mycustomprotocol")
server.registerThreadInitializer(initThread)
server.serve()