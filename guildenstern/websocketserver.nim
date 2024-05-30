## Websocket server
## 
## Example
## =======
##
## .. code-block:: Nim
##
##  import locks
##  import guildenstern/[dispatcher, websocketserver]
##  
##  var lock: Lock
##  initLock(lock)
##  var sockets = newSeq[SocketHandle]()
##  
##  proc onUpgrade(): bool =
##    {.gcsafe.}:
##      withLock(lock):
##        sockets.add(ws.socketdata.socket)
##    true
##    
##  proc afterUpgrade() =
##    wsserver.send(ws.socketdata.socket, "websocket connected!")
##  
##  proc onMessage() =
##    {.gcsafe.}:
##      withLock(lock):
##        let reply = getMessage()
##        discard wsserver.send(sockets, reply)
##  
##  proc onClose(socketdata: ptr SocketData, cause: SocketCloseCause, msg: string) =
##    {.gcsafe.}:
##      withLock(lock):
##        let index = sockets.find(socketdata.socket)
##        sockets.del(index)
##         
##  proc onRequest() =
##    let html = """<!doctype html><title>Web socket chat</title>
##    <script>
##      let websocket = new WebSocket("ws://" + location.host.slice(0, -1) + '1')
##      websocket.onmessage = function(evt) {
##      document.getElementById("ul").appendChild(document.createElement("li")).innerHTML = evt.data }
##    </script>
##    <body><form action="" onsubmit="return false"><input id="input">
##    <button type=submit onclick="websocket.send(getElementById('input').value+'.')">Send</button>
##    </form><ul id="ul"></ul></body>"""
##    reply(Http200, html)
##  
##  let server = newWebsocketServer(onUpgrade, afterUpgrade, onMessage, onClose)
##  server.start(5051)
##  let httpserver = newHttpServer(onRequest)
##  httpserver.start(5050)
##  joinThreads(server.thread, httpserver.thread)
##  deinitLock(lock)
##  

import nativesockets, net, posix, os, base64, times, std/monotimes, sets, locks
when not defined(nimdoc): from sha import secureHash, `$`
export posix.SocketHandle

import httpserver
export httpserver


type
  Opcode* = enum
    Cont = 0x0                ## continuation frame
    Text = 0x1                ## text frame
    Binary = 0x2              ## binary frame
    Close = 0x8               ## connection close
    Ping = 0x9                ## ping
    Pong = 0xa                ## pong
    WsFail = 0xe                ## protocol failure / connection lost in flight
  
  SendState = enum NotStarted, Continue, Delivered, Err

  WsUpgradeCallback* = proc(): bool {.gcsafe, nimcall, raises: [].}
  WsAfterUpgradeCallback* = proc() {.gcsafe, nimcall, raises: [].}
  WsMessageCallback* = proc() {.gcsafe, nimcall, raises: [].}

  WebsocketServer* = ref object of HttpServer
    upgradeCallback*: WsUpgradeCallback
    afterUpgradeCallback*: WsAfterUpgradeCallback
    messageCallback*: WsMessageCallback
    sendingsockets: HashSet[posix.Sockethandle]
    sendlock: Lock

  WebsocketContext* = ref object of HttpContext
    opcode*: OpCode

  State = tuple[sent: int, sendstate: SendState]
  
  WsDelivery* = tuple[sockets: seq[posix.SocketHandle], message: ptr string, binary: bool, states: seq[State]]
    ## `send` takes pointer to this as parameter.
    ## | `sockets`: the websockets that should receive this message
    ## | `message`: the message to send
    ## | `binary`: whether the message contains bytes or chars
  

const MaxParallelSendingSockets {.intdefine.} = 500

var
  wsresponseheader {.threadvar.}: string
  maskkey {.threadvar.}: array[4, char]
  delivery {.threadvar.}: WsDelivery

  
{.push checks: off.}

proc isWebsocketContext*(): bool = return socketcontext is WebsocketContext

template ws*(): untyped =
  ## Casts the socketcontext thread local variable into a WebsocketContext
  WebsocketContext(socketcontext)


template wsserver*(): untyped =
  ## Casts the socketcontext.socketdata.server into a WebsocketServer
  WebsocketServer(socketcontext.socketdata.server)


when not defined(nimdoc):
  proc `$`(x: SocketHandle): string {.inline.} = $(x.cint)

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
    closeSocket(ProtocolViolated, msg)
    ws.opcode = WsFail
    return -1


  proc bytesRecv(fd: posix.SocketHandle, buffer: ptr char, size: int): int =
    return recv(fd, buffer, size, 0)

  {.push warnings: off.} # HoleEnumConv
  proc recvHeader(): int =
    if ws.socketdata.socket.bytesRecv(ws.request[0].addr, 2) != 2: error("no data")
    let b0 = ws.request[0].uint8
    let b1 = ws.request[1].uint8
    ws.opcode = (b0 and 0x0f).Opcode
    if b0[1] or b0[2] or b0[3]: error("protocol")
    var expectedLen: int = 0

    let headerLen = uint(b1 and 0x7f)
    if headerLen == 0x7e:
      var lenstrlen = ws.socketdata.socket.bytesRecv(ws.request[0].addr, 2)
      if lenstrlen != 2: error("length")    
      expectedLen = nativesockets.htons(cast[ptr uint16](ws.request[0].addr)[]).int
    elif headerLen == 0x7f:
      var lenstrlen = ws.socketdata.socket.bytesRecv(ws.request[0].addr, 8)
      if lenstrlen != 8: error("length")
    else: expectedLen = headerLen.int

    let maskKeylen = ws.socketdata.socket.bytesRecv(maskkey[0].addr, 4)
    if maskKeylen != 4: error("length")

    if expectedLen > server.bufferlength: error("Maximum request size bound to be exceeded: " & $(expectedLen))
    
    return expectedLen
  {.pop.}

  proc recvFrame() =
    var expectedlen: int  
    expectedlen = recvHeader()
    if ws.opcode in [WsFail, Close]:
      if ws.opcode == Close: closeSocket(ClosedbyClient, "")
      return
    while true:
      if shuttingdown: (ws.opcode = WsFail; return)
      let ret =
        if ws.requestlen == 0: recv(ws.socketdata.socket, addr ws.request[0], expectedLen.cint, 0x40)
        else: recv(ws.socketdata.socket, addr ws.request[ws.requestlen], (expectedLen - ws.requestlen).cint, 0)
      if shuttingdown: (ws.opcode = WsFail; return)

      if ret == 0: (closeSocket(ClosedbyClient, ""); ws.opcode = WsFail; return)
      if ret == -1:
        let lastError = osLastError().int
        let cause =
          if lasterror in [2,9]: AlreadyClosed
          elif lasterror == 32: ConnectionLost
          elif lasterror == 104: ClosedbyClient
          else: NetErrored
        wsserver.log(WARN, "websocket " & $ws.socketdata.socket & " receive error: " & $lastError & " " & osErrorMsg(OSErrorCode(lastError)))
        ws.opcode = WsFail
        closeSocket(cause, "ws receive error")
        return

      ws.requestlen += ret
      if ws.requestlen == expectedlen: return


  proc receiveWs() =
    ws.requestlen = 0
    try:
      recvFrame()
      if ws.opcode in [WsFail, Close]: return
      while ws.opcode == Cont: recvFrame()
      for i in 0 ..< ws.requestlen: ws.request[i] = (ws.request[i].uint8 xor maskkey[i mod 4].uint8).char
    except:
      wsserver.log(WARN, "websocket " & $ws.socketdata.socket & " receive exception")
      closeSocket(Excepted, "ws receive exception")
      ws.opcode = WsFail


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


  func swapBytesBuiltin(x: uint64): uint64 {.importc: "__builtin_bswap64", nodecl.}


  proc createWsHeader(len: int, opcode: OpCode) =
    wsresponseheader.setLen(0)

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


  proc replyHandshake(): (SocketCloseCause , string) =
    if not readHeader(): return (ProtocolViolated , "ws header failure")
    if not parseRequestLine(): return (ProtocolViolated , "ws requestline failure")
    let key = ws.headers.getOrDefault("sec-websocket-key")
    if key == "": return (ProtocolViolated , "sec-websocket-key header missing")
    let accept = wsserver.upgradeCallback()
    if not accept:
      wsserver.log(DEBUG, "ws upgrade request was not accepted: " & getRequest())
      reply(Http400)
      sleep(200)
      return (CloseCalled , "not accepted")
    when not defined(nimdoc):
      let 
        sh = secureHash(key & "258EAFA5-E914-47DA-95CA-C5AB0DC85B11")
        acceptKey = base64.encode(decodeBase16($sh))
      reply(Http101, ["Sec-WebSocket-Accept: " & acceptKey, "Connection: Upgrade", "Upgrade: webSocket"])
    (DontClose , "")


  proc handleWsUpgradehandshake() =
    var (state , errormessage) = 
      try: replyHandshake()
      except: (Excepted , getCurrentExceptionMsg())
    if state != DontClose:
      closeSocket(state, errormessage)
      return
    ws.socketdata.flags = 1
    if wsserver.afterupgradeCallback != nil:
      wsserver.afterupgradeCallback()


  proc prepareWebsocketContext*(socketdata: ptr SocketData) {.inline.} =
    if (unlikely)socketcontext == nil:
      socketcontext = new WebsocketContext
      delivery = (sockets: newSeq[posix.SocketHandle](), message: nil, binary: false, states: newSeq[State]())
    prepareHttpContext(socketdata)


proc getMessage*(): string =
  return ws.request[0 ..< ws.requestlen]


proc isMessage*(message: string): bool =
  if ws.requestlen != message.len: return false
  for i in countup(0, ws.requestlen - 1):
    if ws.request[i] != ws.request[i]: return false
  true


when not defined(nimdoc):
  proc sendNonblocking(server: WebsocketServer, socket: posix.SocketHandle, text: ptr string, sent: int = 0): (SendState , int) =
    server.log(DEBUG, "writeToWebSocket " & $socket.int & ": " & text[])
    if socket.int in [0, INVALID_SOCKET.int]: return (Err , 0)
    let len = text[].len
    if sent == len: return (Delivered , 0)
    let ret = send(socket, addr text[sent], len - sent, MSG_DONTWAIT)
    if ret < 1:
      if ret == -1:
        let lastError = osLastError().int
        if lastError in [EAGAIN.int, EWOULDBLOCK.int]: return (Continue , 0)
        let cause =
          if lasterror in [2,9]: AlreadyClosed
          elif lasterror == 32: ConnectionLost
          elif lasterror == 104: ClosedbyClient
          else: NetErrored
        server.log(NOTICE, "websocket " & $socket.int & " send error: " & $lastError & " " & osErrorMsg(OSErrorCode(lastError)))
        server.closeOtherSocket(socket, cause, osErrorMsg(OSErrorCode(lastError)))
      elif ret < -1:
        server.log(NOTICE, "websocket " & $socket.int & " send error")
        server.closeOtherSocket(socket, Excepted, getCurrentExceptionMsg())
      return (Err , 0)
    if sent + ret == len: return (Delivered , ret)
    return (Continue , ret)


  proc closeSocketsInFlight(server: WebsocketServer, sockets: seq[posix.SocketHandle], states: seq[State]): int =
    for i in 0 ..< states.len:
      if states[i].sendstate == Continue:
        server.closeOtherSocket(sockets[i], TimedOut, "")
        withLock(server.sendlock): server.sendingsockets.excl(sockets[i])
        result.inc


proc send*(server: GuildenServer, delivery: ptr WsDelivery, timeoutsecs = 10, sleepmillisecs = 10): int {.discardable.} =
  ## Sends message to multiple websockets at once. Uses non-blocking I/O so that slow receivers do not slow down fast receivers.
  ## | Can be called from multiple threads in parallel.
  ## | `timeoutsecs`: a timeout after which sending is given up and all sockets with messages in-flight are closed
  ## | `sleepmillisecs`: if all in-flight receivers are blocking, will sleep for (sleepmillisecs * in-flight receiver count) milliseconds
  ## Returns amount of websockets that had to be closed
  when defined(nimdoc): discard
  else:
    if delivery.sockets.len == 0: return
    let timeout = initDuration(seconds = timeoutsecs)
    let start = getMonoTime()
    server.log(TRACE, "starts sending websockets")
    if delivery.message[].len == 0:
      delivery.message[] = "PING"
      createWsHeader(4, Opcode.Pong)
    else:
      if delivery.binary: createWsHeader(delivery.message[].len, OpCode.Binary) else: createWsHeader(delivery.message[].len, Opcode.Text)
    var handled = 0
    delivery.states.setLen(delivery.sockets.len)
    for i in 0 ..< delivery.states.len: delivery.states[i] = (0, NotStarted)
    var s = -1
    var blockedsockets = 0
    while true:
      if shuttingdown: return -1
      if getMonoTime() - start > timeout:
        result += WebsocketServer(server).closeSocketsInFlight(delivery.sockets, delivery.states)
        server.log(NOTICE, "send timed out")
        return result
      s.inc
      if s == delivery.sockets.len:
        s = -1
        continue
      if delivery.states[s].sendstate in [Delivered, Err]: continue
      var ret: int

      if delivery.states[s].sendstate == NotStarted:
        withLock(WebsocketServer(server).sendlock):
          if WebsocketServer(server).sendingsockets.len == MaxParallelSendingSockets:
            server.log(INFO, "MaxParallelSendingSockets reached, sleeping for " & $(sleepmillisecs) & " ms")
            suspend(sleepmillisecs)
            continue
          if WebsocketServer(server).sendingsockets.contains(delivery.sockets[s]): continue
          WebsocketServer(server).sendingsockets.incl(delivery.sockets[s])
        var headerstate: SendState
        (headerstate , ret) = WebsocketServer(server).sendNonblocking(delivery.sockets[s], addr wsresponseheader)
        if headerstate == Delivered: delivery.states[s].sendstate = Continue # only header delivered
        else:
          delivery.states[s].sendstate = Err
          withLock(WebsocketServer(server).sendlock):
            WebsocketServer(server).sendingsockets.excl(delivery.sockets[s])
          if headerstate == Continue: server.closeOtherSocket(delivery.sockets[s], TimedOut)
          result.inc
          handled.inc
          server.log(TRACE, "wsockets processed: " & $handled & "/" & $delivery.sockets.len)
          if handled == delivery.sockets.len: break
        continue
        
      (delivery.states[s].sendstate , ret) = WebsocketServer(server).sendNonblocking(delivery.sockets[s], delivery.message, delivery.states[s].sent)
      case delivery.states[s].sendstate
        of Err: result.inc
        of Continue:
          delivery.states[s].sent += ret      
          blockedsockets.inc
          if blockedsockets >= delivery.sockets.len - handled:           
            server.log(DEBUG, "all remaining websockets are blocking, suspending for " & $(blockedsockets * sleepmillisecs) & " ms")
            suspend(blockedsockets * sleepmillisecs)
            blockedsockets = 0
        else: discard
      if delivery.states[s].sendstate != Continue:
        withLock(WebsocketServer(server).sendlock): WebsocketServer(server).sendingsockets.excl(delivery.sockets[s])
        handled.inc
        server.log(TRACE, "wsockets processed: " & $handled & "/" & $delivery.sockets.len)
        if handled >= delivery.sockets.len: break
    server.log(TRACE, "websocket send finished")


proc send*(server: GuildenServer, sockets: seq[posix.SocketHandle], message: string, timeoutsecs = 20, sleepmillisecs = 100): bool {.discardable.} =
  when compiles(unsafeAddr message):
    when not defined(nimdoc) and not defined(gcDestructors): {.fatal: "mm:arc or mm:orc required".}
    delivery.sockets = sockets
    delivery.message = unsafeAddr(message)
    delivery.binary = false
    return send(WebsocketServer(server), unsafeAddr delivery, timeoutsecs, sleepmillisecs) == 0
  else:  {.fatal: "posix.send requires taking pointer to message, but message has no address".}


proc send*(server: GuildenServer, socket: posix.SocketHandle, message: string, timeoutsecs = 20, sleepmillisecs = 100): bool {.discardable.} =
  when compiles(unsafeAddr message):
    when not defined(nimdoc) and not defined(gcDestructors): {.fatal: "mm:arc or mm:orc required".}
    delivery.sockets.setLen(1)
    delivery.sockets[0] = socket
    delivery.message = unsafeAddr(message)
    delivery.binary = false
    return send(WebsocketServer(server), unsafeAddr delivery, timeoutsecs, sleepmillisecs) == 0
  else: {.fatal: "posix.send requires taking pointer to message, but message has no address".}


proc handleWsRequest(data: ptr SocketData) {.gcsafe, nimcall, raises: [].} =
    prepareWebsocketContext(data)
    if (unlikely)ws.socketdata.flags == 0:
      handleWsUpgradehandshake()
      return
    receiveWs()
    if unlikely(ws.opcode in [WsFail, Close]): return
    if ws.opcode != Ping: {.gcsafe.}: wsserver.messageCallback()
    else:
      let pingmsg = ""
      if not server.send(ws.socketdata.socket, pingmsg, 5):
        server.log(NOTICE, "websocket " & $ws.socketdata.socket & " blocking, could not autoreply to Ping")


proc newWebsocketServer*(upgradecallback: WsUpgradeCallback, afterupgradecallback: WsAfterUpgradeCallback,
 onwsmessagecallback: WsMessageCallback, onclosesocketcallback: OnCloseSocketCallback, loglevel = LogLevel.WARN): WebsocketServer =
  result = new WebsocketServer
  when not defined(nimdoc):
    initHttpServer(result, loglevel, true, SingleBuffer, ["sec-websocket-key"])
    result.handlerCallback = handleWsRequest
    result.upgradeCallback = upgradecallback
    result.afterupgradeCallback = afterupgradecallback
    result.messageCallback = onwsmessagecallback
    result.onCloseSocketCallback = onclosesocketcallback
    initLock(result.sendlock)
    result.sendingsockets = initHashSet[posix.Sockethandle](3 * MaxParallelSendingSockets)

{.pop.}