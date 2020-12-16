## Websocket handler.
## 
## **Example:**
##
## .. code-block:: Nim
##
##    
##    import nativesockets, locks
##    import guildenstern/[ctxws, ctxheader]
##    
##    let html = """<!doctype html><title>WsCtx</title>
##      <script>
##      let websocket = new WebSocket("ws://" + location.host.slice(0, -1) + '1')
##      websocket.onmessage = function(evt) {
##        document.getElementById("ul").appendChild(document.createElement("li")).innerHTML = evt.data }
##      </script>
##      <body><button onclick="websocket.send('hallo')">say hallo</button>
##      <button onclick="websocket.close()">close</button><ul id="ul">"""
##    
##    var server = new GuildenServer
##    var lock: Lock # playing it safe by always serializing access to mutating globals (socket, in this case)
##    var socket = osInvalidSocket
##    
##    proc onUpgradeRequest(ctx: WsCtx): bool =
##      withLock(lock): socket = ctx.socketdata.socket
##      true
##    
##    proc onMessage(ctx: WsCtx) = echo "client says: ", ctx.getRequest()
##      
##    proc sendMessage() =
##      withLock(lock):
##        if socket != osInvalidSocket:
##          let reply = "hello"
##          discard server.sendWs(socket, reply)
##    
##    proc onLost(gs: ptr GuildenServer, data: ptr SocketData, lostsocket: SocketHandle) =
##      withLock(lock):
##        if lostsocket.int == socket.int:
##          echo "websocket connection lost"
##          socket = osInvalidSocket
##           
##    proc onRequest(ctx: HttpCtx) = ctx.reply(Http200, html)
##    
##    server.initHeaderCtx(onRequest, 5050)
##    server.initWsCtx(onUpgradeRequest, onMessage, 5051)
##    server.registerTimerhandler(sendMessage, 2000)
##    server.registerConnectionlosthandler(onLost)
##    echo "Point your browser to localhost:5050"
##    initLock(lock)
##    server.serve()
##    deinitLock(lock)

import nativesockets, net, posix, os, std/sha1, base64
from httpcore import Http101

when not defined(nimdoc):
  import guildenstern
  export guildenstern
else:
  import guildenserver, ctxhttp

from ctxheader import receiveHeader


const MaxWsRequestLength* {.intdefine.} = 100000

type
  Opcode* = enum
    Cont = 0x0                ## continuation frame
    Text = 0x1                ## text frame
    Binary = 0x2              ## binary frame
    Close = 0x8               ## connection close
    Ping = 0x9                ## ping
    Pong = 0xa                ## pong
    Fail = 0xe                ## protocol failure / connection lost in flight

  WsCtx* = ref object of HttpCtx
    opcode*: OpCode

  WsUpgradeRequestCallback =  proc(ctx: WsCtx): bool {.gcsafe, nimcall, raises: [].}
  WsMessageCallback = proc(ctx: WsCtx){.gcsafe, nimcall, raises: [].}

var
  WsCtxId: CtxId
  upgraderequestcallback: WsUpgradeRequestCallback
  messageCallback: WsMessageCallback
  
  wsresponseheader {.threadvar.}: string
  ctx {.threadvar.}: WsCtx 
  maskkey {.threadvar.}: array[4, char]
  
{.push checks: off.}

template `[]`(value: uint8, index: int): bool =
  ## Get bits from uint8, uint8[2] gets 2nd bit.
  (value and (1 shl (7 - index))) != 0

  #[
   0                   1                   2                   3
   0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
  +-+-+-+-+-------+-+-------------+-------------------------------+
  |F|R|R|R| opcode|M| Payload len |    Extended payload length    |
  |I|S|S|S|  (4)  |A|     (7)     |             (16/64)           |
  |N|V|V|V|       |S|             |   (if payload len==126/127)   |
  | |1|2|3|       |K|             |                               |
  +-+-+-+-+-------+-+-------------+ - - - - - - - - - - - - - - - +
  |     Extended payload length continued, if payload len == 127  |
  + - - - - - - - - - - - - - - - +-------------------------------+
  |                               |Masking-key, if MASK set to 1  |
  +-------------------------------+-------------------------------+
  | Masking-key (continued)       |          Payload Data         |
  +-------------------------------- - - - - - - - - - - - - - - - +
  :                     Payload Data continued ...                :
  + - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +
  |                     Payload Data continued ...                |
  +---------------------------------------------------------------+]#


template error(msg: string) =
  let errormsg = "websocket " & $ctx.socketdata.socket & " fail: " & msg
  ctx.gs.errormsg &= " | " & errormsg
  when defined(fulldebug): echo errormsg
  ctx.closeSocket(ProtocolViolated)
  ctx.opcode = Fail
  return -1

proc bytesRecv(fd: posix.SocketHandle, buffer: ptr char, size: int): int =
  return recv(fd, buffer, size, 0)


proc recvHeader(): int =
  if posix.SocketHandle(ctx.socketdata.socket).bytesRecv(request[0].addr, 2) != 2: error("no data")
  let b0 = request[0].uint8
  let b1 = request[1].uint8
  ctx.opcode = (b0 and 0x0f).Opcode
  if b0[1] or b0[2] or b0[3]: error("protocol")
  var expectedLen: int = 0

  let headerLen = uint(b1 and 0x7f)
  if headerLen == 0x7e:
    var lenstrlen = posix.SocketHandle(ctx.socketdata.socket).bytesRecv(request[0].addr, 2)
    if lenstrlen != 2: error("length")    
    expectedLen = nativesockets.htons(cast[ptr uint16](request[0].addr)[]).int
  elif headerLen == 0x7f:
    var lenstrlen = posix.SocketHandle(ctx.socketdata.socket).bytesRecv(request[0].addr, 8)
    if lenstrlen != 8: error("length")
  else: expectedLen = headerLen.int

  let maskKeylen = posix.SocketHandle(ctx.socketdata.socket).bytesRecv(maskkey[0].addr, 4)
  if maskKeylen != 4: error("length")

  if expectedLen > MaxWsRequestLength: error("Maximum request size bound to be exceeded: " & $(expectedLen))
  
  return expectedLen


proc recvFrame() =
  var expectedlen: int  
  expectedlen = recvHeader()
  if ctx.opcode in [Fail, Close]:
    if ctx.opcode == Close: ctx.closeSocket(ClosedbyClient)
    return
  while true:
    if shuttingdown: (ctx.opcode = Fail; return)
    let ret =
      if ctx.requestlen == 0: recv(ctx.socketdata.socket, addr request[0], expectedLen.cint, 0x40)
      else: recv(ctx.socketdata.socket, addr request[ctx.requestlen], (expectedLen - ctx.requestlen).cint, 0)
    if shuttingdown: (ctx.opcode = Fail; return)

    if ret == 0: (ctx.closeSocket(ClosedbyClient); ctx.opcode = Fail; return)
    if ret == -1:
      let lastError = osLastError().int
      let cause =
        if lasterror in [2,9]: AlreadyClosed
        elif lasterror == 32: ConnectionLost
        elif lasterror == 104: ClosedbyClient
        else: NetErrored
      when defined(fulldebug): echo "websocket " & $ctx.socketdata.socket & " receive error: " & $lastError & " " & osErrorMsg(OSErrorCode(lastError))
      ctx.gs.errormsg &= " | " & osErrorMsg(OSErrorCode(lastError))
      ctx.opcode = Fail
      ctx.closeSocket(cause)
      return

    ctx.requestlen += ret
    if ctx.requestlen == expectedlen: return


proc receiveWs() =
  ctx.requestlen = 0
  try:
    recvFrame()
    if ctx.opcode in [Fail, Close]: return
    while ctx.opcode == Cont: recvFrame()
    for i in 0 ..< ctx.requestlen: request[i] = (request[i].uint8 xor maskkey[i mod 4].uint8).char
  except:
    ctx.gs.errormsg &= " | " & getCurrentExceptionMsg()
    ctx.closeSocket(Excepted)
    ctx.opcode = Fail


proc nibbleFromChar(c: char): int =
  case c:
    of '0'..'9': (ord(c) - ord('0'))
    of 'a'..'f': (ord(c) - ord('a') + 10)
    of 'A'..'F': (ord(c) - ord('A') + 10)
    else: 255


proc decodeBase16(str: string): string =
  result = newString(str.len div 2)
  for i in 0 ..< result.len:
    result[i] = chr(
      (nibbleFromChar(str[2 * i]) shl 4) or
      nibbleFromChar(str[2 * i + 1]))

proc replyHandshake(): bool =
  if not ctx.receiveHeader(): return false
  var headers = [""]  
  ctx.parseHeaders(["sec-websocket-key"], headers)
  if headers[0] == "": return false
  if not ctx.upgraderequestcallback(): return false
  let 
    sh = secureHash(headers[0] & "258EAFA5-E914-47DA-95CA-C5AB0DC85B11")
    acceptKey = base64.encode(decodeBase16($sh))
  ctx.reply(Http101, ["Sec-WebSocket-Accept: " & acceptKey, "Connection: Upgrade", "Upgrade: webSocket"])
  true

proc handleWsUpgradehandshake(gs: ptr GuildenServer, data: ptr SocketData) {.gcsafe, nimcall, raises: [].} =
  if ctx == nil:
    ctx = new WsCtx
    initHttpCtx(ctx, gs, data)
    if request.len < MaxWsRequestLength + 1: request = newString(MaxWsRequestLength + 1)
  ctx.gs = gs
  ctx.socketdata = data
  ctx.requestlen = 0
  if replyHandshake(): data.ctxid = WsCtxId
  else:
    ctx.reply(Http204)
    sleep(3000)
    ctx.closeSocket(ProtocolViolated)


proc handleWsMessage(gs: ptr GuildenServer, data: ptr SocketData) {.gcsafe, nimcall, raises: [].} =
  if ctx == nil:
    ctx = new WsCtx
    initHttpCtx(ctx, gs, data)
    if request.len < MaxWsRequestLength + 1: request = newString(MaxWsRequestLength + 1)
  ctx.gs = gs
  ctx.socketdata = data
  ctx.requestlen = 0
  receiveWs()
  if ctx.opcode notin [Fail, Close]:
    {.gcsafe.}: messageCallback(ctx)
  

proc initWsCtx*(gs: var GuildenServer, onwsupgraderequestcallback: WsUpgradeRequestCallback, onwsmessage: WsMessageCallback, port: int) =
  {.gcsafe.}:
    upgraderequestcallback = onwsupgraderequestcallback
    messageCallback = onwsmessage
    discard gs.registerHandler(handleWsUpgradehandshake, port)
    WsCtxId = gs.registerHandler(handleWsMessage, -1)


proc send(gs: ptr GuildenServer, socket: posix.SocketHandle, text: ptr string, length: int = -1): bool =
  let len = if length == -1: text[].len else: length
  var sent = 0
  while sent < len:
    if shuttingdown: return false    
    let ret = send(socket, addr text[sent], len - sent, 0)
    if ret < 1:
      if ret == -1:
        let lastError = osLastError().int
        let cause =
          if lasterror in [2,9]: AlreadyClosed
          elif lasterror == 32: ConnectionLost
          elif lasterror == 104: ClosedbyClient
          else: NetErrored
        when defined(fulldebug): echo "websocket " & $ctx.socketdata.socket & " send error: " & $lastError & " " & osErrorMsg(OSErrorCode(lastError))
        ctx.gs.errormsg &= " | " & osErrorMsg(OSErrorCode(lastError))
        ctx.closeSocket(cause)
      elif ret < -1:
        ctx.gs.errormsg &= " | " & getCurrentExceptionMsg()
        ctx.closeSocket(Excepted)
      return false
    sent.inc(ret)
    if sent == len: return true
  

proc createWsHeader(len: int, binary = false) =
  wsresponseheader = ""
  var b0 = if binary: (0x2.uint8 and 0x0f) else: (0x1.uint8 and 0x0f)
  b0 = b0 or 128u8

  var b1 = 0u8
  if len <= 125: b1 = len.uint8
  elif len > 125 and len <= 0xffff: b1 = 126u8
  else: b1 = 127u8

  wsresponseheader.add(b0.char)
  wsresponseheader.add(b1.char)
    
  if len > 125 and len <= 0xffff:
    wsresponseheader.add($nativesockets.htons(len.uint16))
  elif len > 0xffff:
    wsresponseheader.add char((len shr 56) and 255)
    wsresponseheader.add char((len shr 48) and 255)
    wsresponseheader.add char((len shr 40) and 255)
    wsresponseheader.add char((len shr 32) and 255)
    wsresponseheader.add char((len shr 24) and 255)
    wsresponseheader.add char((len shr 16) and 255)
    wsresponseheader.add char((len shr 8) and 255)
    wsresponseheader.add char(len and 255)


proc sendWs*(gs: GuildenServer, socket: nativesockets.SocketHandle, message: ptr string, length: int = -1, binary = false): bool =
  if length == 0 or message == nil: return
  let len = if length == -1: message[].len else: length
  createWsHeader(len, binary)
  if send(unsafeAddr gs, posix.SocketHandle(socket), addr wsresponseheader): return send(unsafeAddr gs, posix.SocketHandle(socket), message, len)

template sendWs*(gs: GuildenServer, socket: nativesockets.SocketHandle, message: string, length: int = -1, binary = false): bool =
  when compiles(unsafeAddr message):
    sendWs(gs, socket, unsafeAddr message, length, binary)
  else:  {.fatal: "posix.send requires taking pointer to message, but message has no address".}


proc replyWs*(ctx: Ctx, text: ptr string, length = -1, binary = false): bool {.inline.} =
  return ctx.gs[].sendWs(ctx.socketdata.socket, text, length, binary)

template replyWs*(ctx: Ctx, message: string, length = -1, binary = false): bool =
  when compiles(unsafeAddr message):
    replyWs(ctx, unsafeAddr message, length, binary) 
  else:  {.fatal: "posix.send requires taking pointer to message, but message has no address".}

{.pop.}