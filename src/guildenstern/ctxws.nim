## Websocket handler.
## 
## **Example:**
##
## .. code-block:: Nim
##
##    
##    import locks
##    import guildenstern/[ctxws, ctxheader, ctxtimer]
##    
##    var server = new GuildenServer
##    var lock: Lock # serializing access to mutating globals is usually good idea (socket, in this case)
##    var socket = INVALID_SOCKET
##    
##    proc onUpgradeRequest(ctx: WsCtx): (bool , proc()) =
##      {.gcsafe.}: withLock(lock): socket = ctx.socketdata.socket
##      (true , proc() = ctx.sendWs("request accepted!"))
##    
##    proc onMessage(ctx: WsCtx) = echo "client says: ", ctx.getRequest()
##      
##    proc sendMessage() =
##      withLock(lock):
##        if socket != INVALID_SOCKET:
##          let reply = "hello"
##          discard server.sendWs(socket, reply)
##    
##    proc onLost(ctx: Ctx, lostsocket: SocketHandle, cause: SocketCloseCause, msg: string) =
##      withLock(lock):
##        if lostsocket == socket:
##          echo cause
##          socket = INVALID_SOCKET
##           
##    proc onRequest(ctx: HttpCtx) =
##      let html = """<!doctype html><title>WsCtx</title>
##      <script>
##        let websocket = new WebSocket("ws://" + location.host.slice(0, -1) + '1')
##        websocket.onmessage = function(evt) {
##        document.getElementById("ul").appendChild(document.createElement("li")).innerHTML = evt.data }
##      </script>
##      <body><button onclick="websocket.send('hallo')">say hallo</button>
##      <button onclick="websocket.close()">close</button><ul id="ul">""" 
##      ctx.reply(Http200, html)
##    
##    server.initHeaderCtx(onRequest, 5050, false)
##    server.initWsCtx(onUpgradeRequest, onMessage, 5051)
##    server.initTimerCtx(200, sendMessage)
##    server.registerConnectionclosedhandler(onLost)
##    echo "Point your browser to localhost:5050"
##    initLock(lock)
##    server.serve()
##    deinitLock(lock)

import nativesockets, net, posix, os, std/sha1, base64, times, std/monotimes
from httpcore import Http101

from ctxheader import receiveHeader
from dispatcher import getLoads

when not defined(nimdoc):
  import guildenstern
  export guildenstern
else:
  import guildenserver, ctxhttp

type
  Opcode* = enum
    Cont = 0x0                ## continuation frame
    Text = 0x1                ## text frame
    Binary = 0x2              ## binary frame
    Close = 0x8               ## connection close
    Ping = 0x9                ## ping
    Pong = 0xa                ## pong
    Fail = 0xe                ## protocol failure / connection lost in flight
  
  WsUpgradeRequestCallback* =  proc(ctx: WsCtx): (bool , proc() {.closure, raises: [].}) {.gcsafe, nimcall, raises: [].}
  ## This must be set with initWsCtx.
  ## Return tuple with following parameters:
  ## | `bool`: whether to accept the upgrade request or to close the socket
  ## | `proc`: After accepted upgrade request is processed this closure will be called (if assigned)

  WsMessageCallback = proc(ctx: WsCtx) {.gcsafe, nimcall, raises: [].}

  PortDatum = object
    port: uint16
    upgradehandler: WsUpgradeRequestCallback
    messagehandler: WsMessageCallback

  WsCtx* = ref object of HttpCtx
    opcode*: OpCode
  
  WsDelivery* = tuple[sockets: seq[posix.SocketHandle], message: string, length: int, binary: bool]
    ## `multiSendWs` takes an open array of these as parameter.
    ## | `sockets`: the websockets that should receive this message
    ## | `message`: the message to send (note that this not a pointer - a deep copy is used)
    ## | `length`: amount of chars to send from message. If < 1, defaults to message.len
    ## | `binary`: whether the message contains bytes or chars

var
  portdata: array[MaxHandlersPerCtx, PortDatum]
  wsresponseheader {.threadvar.}: string
  ctx {.threadvar.}: WsCtx 
  maskkey {.threadvar.}: array[4, char]
  
{.push checks: off.}

proc at(port: uint16): int {.inline.} =
  while portdata[result].port != port: result += 1


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
  ctx.closeSocket(ProtocolViolated, msg)
  ctx.opcode = Fail
  return -1


proc bytesRecv(fd: posix.SocketHandle, buffer: ptr char, size: int): int =
  return recv(fd, buffer, size, 0)

{.push warning[HoleEnumConv]: off.}
proc recvHeader(): int =
  if ctx.socketdata.socket.bytesRecv(request[0].addr, 2) != 2: error("no data")
  let b0 = request[0].uint8
  let b1 = request[1].uint8
  ctx.opcode = (b0 and 0x0f).Opcode
  if b0[1] or b0[2] or b0[3]: error("protocol")
  var expectedLen: int = 0

  let headerLen = uint(b1 and 0x7f)
  if headerLen == 0x7e:
    var lenstrlen = ctx.socketdata.socket.bytesRecv(request[0].addr, 2)
    if lenstrlen != 2: error("length")    
    expectedLen = nativesockets.htons(cast[ptr uint16](request[0].addr)[]).int
  elif headerLen == 0x7f:
    var lenstrlen = ctx.socketdata.socket.bytesRecv(request[0].addr, 8)
    if lenstrlen != 8: error("length")
  else: expectedLen = headerLen.int

  let maskKeylen = ctx.socketdata.socket.bytesRecv(maskkey[0].addr, 4)
  if maskKeylen != 4: error("length")

  if expectedLen > MaxRequestLength: error("Maximum request size bound to be exceeded: " & $(expectedLen))
  
  return expectedLen
{.pop.}

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
      ctx.gs[].log(WARN, "websocket " & $ctx.socketdata.socket & " receive error: " & $lastError & " " & osErrorMsg(OSErrorCode(lastError)))
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
    ctx.gs[].log(WARN, "websocket " & $ctx.socketdata.socket & " receive exception")
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


proc send(gs: GuildenServer, socket: posix.SocketHandle, text: ptr string, length: int = -1): bool =
  ctx.gs[].log(DEBUG, "writeToWebSocket " & $socket.int & ": " & text[])
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
        ctx.gs[].log(INFO, "websocket " & $socket & " send error: " & $lastError & " " & osErrorMsg(OSErrorCode(lastError)))
        try:
          if socket == ctx.socketdata.socket: ctx.closeSocket(cause, osErrorMsg(OSErrorCode(lastError)))
          else: closeOtherSocket(gs, socket, cause, osErrorMsg(OSErrorCode(lastError)))
        except:
          ctx.gs[].log(NOTICE, "websocket close failed")
      elif ret < -1:
        ctx.gs[].log(INFO, "websocket " & $socket & " send error")
        try:
          if socket == ctx.socketdata.socket: ctx.closeSocket(Excepted, getCurrentExceptionMsg())
          else: closeOtherSocket(gs, socket, Excepted, getCurrentExceptionMsg())
        except:
          ctx.gs[].log(NOTICE, "websocket close failed")
      return false
    sent.inc(ret)
    if sent == len: return true


func swapBytesBuiltin(x: uint64): uint64 {.importc: "__builtin_bswap64", nodecl.}

proc createWsHeader(len: int, opcode: OpCode) =
  wsresponseheader = ""

  # var b0 = if binary: (Opcode.Binary.uint8 and 0x0f) else: (Opcode.Text.uint8 and 0x0f)
  var b0 = opcode.uint8
  b0 = b0 or 128u8
  wsresponseheader.add(b0.char)

  # Payload length can be 7 bits, 7+16 bits, or 7+64 bits.
  # 1st byte: payload len start and mask bit.
  var b1 = 0'u8

  if len <= 125:
    b1 = len.uint8
  elif len > 125 and len <= 0xFFFF:
    b1 = 126'u8
  else:
    b1 = 127'u8
  wsresponseheader.add(b1.char)

  # Only need more bytes if data len is 7+16 bits, or 7+64 bits.
  if len > 125 and len <= 0xFFFF:
    # Data len is 7+16 bits.
    var len16 = len.uint16
    wsresponseheader.add char(((len16 shr 8) and 0xFF).uint8)
    wsresponseheader.add char((len16 and 0xFF).uint8)
  elif len > 0xFFFF:
    # Data len is 7+64 bits.
    var len64 = swapBytesBuiltin(len.uint64) # AMD64 is little endian, Internet is big endian
    for i in 0..<sizeof(len64):
      wsresponseheader.add char((len64 shr (i * 8)) and 0xff)


proc sendWs*(gs: GuildenServer, socket: posix.SocketHandle, message: ptr string, length: int = -1, binary = false): bool {.inline, discardable.} =
  if length == 0 or message == nil or socket == INVALID_SOCKET: return
  let len = if length == -1: message[].len else: length
  if binary: createWsHeader(len, Opcode.Binary) else: createWsHeader(len, OpCode.Text)
  if gs.send(socket, addr wsresponseheader): return gs.send(socket, message, len)


proc sendPong*(gs: GuildenServer, socket: posix.SocketHandle) =
  let message = "PING"
  createWsHeader(4, Opcode.Pong)
  if gs.send(socket, addr wsresponseheader): discard gs.send(socket, unsafeaddr message, 4)


proc replyHandshake(): (SocketCloseCause , string ,  proc() {.closure, raises:[].}) =
  if not ctx.receiveHeader(): return (ProtocolViolated , "header failure" , nil)
  if not ctx.parseRequestLine(): return (ProtocolViolated , "requestline failure" , nil)
  var headers = [""]  
  ctx.parseHeaders(["sec-websocket-key"], headers)
  if headers[0] == "": return (ProtocolViolated , "sec-websocket-key header missing" , nil)
  let (accept , afterupgraderequestcallback) = portdata[at(ctx.socketdata.port)].upgradehandler(ctx)
  if not accept:
    ctx.gs[].log(DEBUG, "ws upgrade request was not accepted: " & ctx.getRequest())
    return (CloseCalled , "" , nil)
  let 
    sh = secureHash(headers[0] & "258EAFA5-E914-47DA-95CA-C5AB0DC85B11")
    acceptKey = base64.encode(decodeBase16($sh))
  ctx.reply(Http101, ["Sec-WebSocket-Accept: " & acceptKey, "Connection: Upgrade", "Upgrade: webSocket"])
  (DontClose , "" , afterupgraderequestcallback)


proc handleWsUpgradehandshake() =
  {.gcsafe.}:
    var (state , errormessage , afterupgraderequestcallback) = 
      try: replyHandshake()
      except: (Excepted , "" , nil)
  if state != DontClose:
    ctx.closeSocket(state, errormessage)
    return
  ctx.socketdata.flags = 1
  if afterupgraderequestcallback != nil:
    {.gcsafe.}: afterupgraderequestcallback()


proc handleWsRequest(gs: ptr GuildenServer, data: ptr SocketData) {.gcsafe, nimcall, raises: [].} =
    if ctx == nil: ctx = new WsCtx
    initHttpCtx(ctx, gs, data)
    if ctx.socketdata.flags == 0: handleWsUpgradehandshake()
    else:
      receiveWs()
      if ctx.opcode notin [Fail, Close]:
        {.gcsafe.}: portdata[at(data.port)].messagehandler(ctx)


proc initWsCtx*(gs: var GuildenServer, onwsupgraderequestcallback: WsUpgradeRequestCallback, onwsmessage: WsMessageCallback, port: int) =
  var index = 0
  while index < MaxHandlersPerCtx and portdata[index].port != 0: index += 1
  if index == MaxHandlersPerCtx: raise newException(Exception, "Cannot register over " & $MaxHandlersPerCtx & " ports per WsCtx")
  portdata[index] = PortDatum(port: port.uint16, upgradehandler: onwsupgraderequestcallback, messagehandler: onwsmessage)
  gs.registerHandler(handleWsRequest, port, "websocket")


proc sendWs*(gs: GuildenServer, socket: posix.SocketHandle, message: string, length: int = -1, binary = false): bool {.discardable.} =
  when compiles(unsafeAddr message):
    return gs.sendWs(socket, unsafeAddr message, length, binary)
  else:  {.fatal: "posix.send requires taking pointer to message, but message has no address".}


const DONTWAIT = 0x40.cint

type SendState = enum NotStarted, Continue, Ok, Error

proc sendNonblocking(gs: GuildenServer, socket: posix.SocketHandle, text: ptr string, sent: int = 0, length: int = -1): (SendState , int) =
  gs.log(DEBUG, "nonblockingwriteToWebSocket " & $socket.int & ": " & text[])
  if socket.int in [0, INVALID_SOCKET.int]: return (Error , 0)
  let len = if length == -1: text[].len else: length
  if sent == len: return (Ok , 0)
  let ret = send(socket, addr text[sent], len - sent, DONTWAIT)
  if ret in [EAGAIN.int, EWOULDBLOCK.int]: return (Continue , 0)
  if ret < 1:
      if ret == -1:
        let lastError = osLastError().int
        let cause =
          if lasterror in [2,9]: AlreadyClosed
          elif lasterror == 32: ConnectionLost
          elif lasterror == 104: ClosedbyClient
          else: NetErrored
        gs.log(NOTICE, "websocket " & $socket & " nonblockingSend error: " & $lastError & " " & osErrorMsg(OSErrorCode(lastError)))
        closeOtherSocket(gs, socket, cause, osErrorMsg(OSErrorCode(lastError)))
      elif ret < -1:
        gs.log(NOTICE, "websocket " & $socket & " nonblockingSend error")
        closeOtherSocket(gs, socket, Excepted, getCurrentExceptionMsg())
      return (Error , 0)
  if sent + ret == len: return (Ok , ret)
  return (Continue , ret)


type State = tuple[sent: int, sendstate: SendState]


proc closeSocketsInFlight(gs: GuildenServer, sockets: seq[posix.SocketHandle], states: seq[State]): int =
  for i in 0 ..< states.len:
    if states[i].sendstate == Continue:
      gs.closeOtherSocket(sockets[i], TimedOut)
      result.inc


proc multiSendWs*(gs: GuildenServer, messages: openArray[WsDelivery], timeoutsecs = 20, sleepmillisecs = 100): int {.discardable.} =
  # TODO: try with io_uring, https://github.com/axboe/liburing
  ## Sends multiple messages to multiple websockets at once. Uses non-blocking I/O so that slow receivers do not slow down fast receivers.
  ## | `timeoutsecs`: a timeout after which sending is given up and all sockets with messages in-flight are closed
  ## | `sleepmillisecs`: if all in-flight receivers are blocking, will sleep for (sleepmillisecs * current thread load) milliseconds
  ## Returns amount of websockets that had to be closed
  if messages.len == 0: return 0
  let timeout = initDuration(seconds = timeoutsecs)
  let start = getMonoTime()
  var m = -1
  while true:
    m.inc
    if m == messages.len: break
    if messages[m].sockets.len == 0 or messages[m].message.len == 0: continue
    let len = if messages[m].length < 1: messages[m].message.len else: messages[m].length
    if messages[m].binary: createWsHeader(len, OpCode.Binary) else: createWsHeader(len, Opcode.Text)
    var handled = 0
    var states = newSeq[State](messages[m].sockets.len)
    for i in 0 ..< states.len: states[i] = (0, NotStarted)
    var s = -1
    var blockedsockets = 0
    while true:
      if shuttingdown: return -1
      if getMonoTime() - start > timeout:
        result += gs.closeSocketsInFlight(messages[m].sockets, states)
        ctx.gs[].log(NOTICE, "multiSendWs timed out")
        return result
      s.inc
      if s == messages[m].sockets.len:
        s = -1
        continue
      if states[s].sendstate in [Ok, Error]: continue
      var ret: int

      if states[s].sendstate == NotStarted:
        var headerstate: SendState
        (headerstate , ret) = gs.sendNonblocking(messages[m].sockets[s], addr wsresponseheader)
        if headerstate == Ok: states[s].sendstate = Continue
        else:
          states[s].sendstate = Error
          if headerstate == Continue: gs.closeOtherSocket(messages[m].sockets[s], TimedOut)
          result.inc
          handled.inc
          if handled == messages[m].sockets.len: break
        continue
        
      (states[s].sendstate , ret) = gs.sendNonblocking(messages[m].sockets[s], unsafeAddr messages[m].message, states[s].sent, len)
      case states[s].sendstate
        of Error: result.inc
        of Continue:
          states[s].sent += ret      
          blockedsockets.inc
          if blockedsockets >= messages[m].sockets.len - handled:
            let currentload: int = getLoads()[0]              
            gs.log(DEBUG, "all remaining multisendws sockets are blocking, sleeping for " & $(currentload * sleepmillisecs) & " ms")
            os.sleep(currentload * sleepmillisecs)
            blockedsockets = 0
        else: discard
      if states[s].sendstate != Continue: 
        handled.inc
        if handled == messages[m].sockets.len: break


proc replyWs*(gs: GuildenServer, ctx: Ctx, text: ptr string, length = -1, binary = false): bool {.inline, discardable.} =
  return gs.sendWs(ctx.socketdata.socket, text, length, binary)


template replyWs*(ctx: Ctx, message: string, length = -1, binary = false): bool =
  when compiles(unsafeAddr message):
    replyWs(ctx, unsafeAddr message, length, binary) 
  else:  {.fatal: "posix.send requires taking pointer to message, but message has no address".}


proc sendWs*(ctx: Ctx, message: string, length = -1, binary = false): bool {.inline, discardable.} =
  when compiles(unsafeAddr message):
    return ctx.gs[].sendWs(ctx.socketdata.socket, unsafeAddr message, length, binary)
  else:  {.fatal: "posix.send requires taking pointer to message, but message has no address".}

{.pop.}