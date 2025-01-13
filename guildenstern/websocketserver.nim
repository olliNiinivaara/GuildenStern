## Websocket server
## 
## see examples/websockettest.nim for a concrete example.
##

import net, os, base64, times, sets, locks
from posix import SocketHandle, send, recv, EAGAIN, EWOULDBLOCK,
 Timespec, clock_gettime, CLOCK_MONOTONIC
import std/importutils # MonoTime.ticks
import std/monotimes
from nativesockets import htons
from std/strutils import startsWith
from std/bitops import setBit
when not defined(nimdoc):
  from sha import secureHash, `$`
export posix.SocketHandle

import httpserver
export httpserver

type
  Opcode = enum
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
    isclient* = false
    upgradeCallback*: WsUpgradeCallback ## Return true to accept, false to decline
    afterUpgradeCallback*: WsAfterUpgradeCallback ## Optional, good for initialization purposes
    messageCallback*: WsMessageCallback ## Triggered when a message is received. Streaming reads are not supported: message length must be shorter than buffersize.

  State = tuple[sent: int, sendstate: SendState]
  
  WsDelivery* = tuple[sockets: seq[posix.SocketHandle], message: ptr string, binary: bool, states: seq[State]]
    ## `send` takes pointer to this as parameter.
    ## | `sockets`: the websockets that should receive this message
    ## | `message`: the message to send (empty message sends a Pong)
    ## | `binary`: whether the message contains bytes or chars
  

const
  MaxParallelSendingSockets {.intdefine.} = 500
  randommaskkey = "aaaa" # used by websocketclient (not very random, though)

let
  MagicString = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
  MagicClose = "close-258EAFA5-E914:"

var
  sendingsockets: HashSet[posix.Sockethandle]
  sendlock: Lock

  opcode {.threadvar.} : OpCode
  wsresponseheader {.threadvar.} : string
  maskkey {.threadvar.} : string
  ismasked {.threadvar.} : bool
  deli {.threadvar.} : WsDelivery

initLock(sendlock)
sendingsockets = initHashSet[posix.Sockethandle](3 * MaxParallelSendingSockets)

when not defined(debug):
  {.push checks: off.}

template ws*(): untyped =
  HttpContext(socketcontext)

template wsserver*(): untyped =
  ## Casts the socketcontext.server into a WebsocketServer
  WebsocketServer(socketcontext.server)

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
  closeSocket(wsserver, thesocket, ProtocolViolated, msg)
  opcode = WsFail
  return -1


proc isTimeout(backoff, totalbackoff: var int): bool {.inline.} =
  suspend(backoff)
  totalbackoff += backoff
  if totalbackoff > wsserver.sockettimeoutms:
    closeSocket(wsserver, thesocket, TimedOut, "websocket time out")
    return true
  backoff *= 2
  return false


proc bytesRecv(buffer: var string, size: int): int =
  if wsserver.sockettimeoutms == -1: return recv(thesocket, addr buffer[0], size, 0)
  var 
    received = 0
    backoff = initialbackoff
    totalbackoff = 0
  while true:
    let res = recv(thesocket, addr buffer[received], size, MSG_DONTWAIT)
    if res == -1:
      let lastError = osLastError().int
      if lastError in [EAGAIN.int, EWOULDBLOCK.int]:
        if not isTimeout(backoff, totalbackoff): continue
        else: error("websocket " & $thesocket & " sent nothing")
      else: error("websocket " & $thesocket & " ws header receive error: " & $lastError & " " & osErrorMsg(OSErrorCode(lastError)))
    elif res > 0:
      received += res
      if received == size: return received
    else: error("websocket " & $thesocket & " ws header receive error")


{.push warnings: off.} # HoleEnumConv
proc recvHeader(): int =
  if bytesRecv(ws.request, 2) != 2: return
  let b0 = ws.request[0].uint8
  let b1 = ws.request[1].uint8
  opcode = (b0 and 0x0f).Opcode
  if b0[1] or b0[2] or b0[3]: error("ws receive protocol error")
  var expectedLen: int = 0

  ismasked = b1[0].uint8 == 1

  let headerLen = uint(b1 and 0x7f)
  if headerLen == 0x7e:
    var lenstrlen = bytesRecv(ws.request, 2)
    if lenstrlen != 2: return  
    expectedLen = nativesockets.htons(cast[ptr uint16](ws.request[0].addr)[]).int
  elif headerLen == 0x7f:
    var lenstrlen = bytesRecv(ws.request, 8)
    if lenstrlen != 8: return
  else: expectedLen = headerLen.int

  if ismasked:
    let maskKeylen = bytesRecv(maskkey, 4)
    if maskKeylen != 4: return
  
  return expectedLen
{.pop.}


proc recvFrame() =
  var expectedlen = recvHeader()
  if opcode == WsFail or expectedlen == 0: return
  if expectedLen > wsserver.bufferlength:
    opcode = WsFail
    closeSocket(wsserver, thesocket, ProtocolViolated, "client tried to offer more data than fits to buffer")
    return
  var
    backoff = initialbackoff
    totalbackoff = 0
  while true:
    if shuttingdown: (opcode = WsFail; return)
    let ret = recv(thesocket, addr ws.request[ws.requestlen], (expectedLen - ws.requestlen).cint, MSG_DONTWAIT)
    if shuttingdown: (opcode = WsFail; return)

    if ret == 0:
      closeSocket(wsserver, thesocket, ClosedbyClient, "websocket " & $thesocket & " receive error")
      opcode = WsFail
      return
    elif ret == -1:
      let lastError = osLastError().int
      if lastError in [EAGAIN.int, EWOULDBLOCK.int]:
         if not isTimeout(backoff, totalbackoff): continue
         else:
          opcode = WsFail
          return
      let cause =
        if lasterror in [2,9]: AlreadyClosed
        elif lasterror == 14: EFault
        elif lasterror == 32: ConnectionLost
        elif lasterror == 104: ClosedbyClient
        else: NetErrored
      opcode = WsFail
      closeSocket(wsserver, thesocket, cause, "ws receive error " & $lastError & ": " & osErrorMsg(OSErrorCode(lastError)))
      return
    else:
      ws.requestlen += ret
      if ws.requestlen == expectedlen: return


proc receiveWs() =
  ws.requestlen = 0
  try:
    recvFrame()
    if opcode == WsFail: return
    while opcode == Cont: recvFrame()
    if ismasked:
      for i in 0 ..< ws.requestlen:
        ws.request[i] = (ws.request[i].uint8 xor maskkey[i mod 4].uint8).char
  except:
    closeSocket(wsserver, thesocket, Excepted, "ws receive exception")
    opcode = WsFail


proc maskMessage(delivery: ptr WsDelivery) =
  for i in 0 ..< delivery.message[].len: delivery.message[i] = (delivery.message[i].uint8 xor randommaskkey[i mod 4].uint8).char


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


proc createWsHeader(len: int, code: OpCode, isclient: bool) =
  wsresponseheader.setLen(0)

  # var b0 = if binary: (Opcode.Binary.uint8 and 0x0f) else: (Opcode.Text.uint8 and 0x0f)
  var b0 = code.uint8
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

  if isclient: b1.setBit(7'u8) # mask bit

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

  if isclient: wsresponseheader.add(randommaskkey) # 4-byte mask


proc replyHandshake(): (SocketCloseCause , string) =
  if not readHeader(): return (AlreadyClosed, "")
  if not parseRequestLine(): return (AlreadyClosed, "")
  let key = ws.headers.getOrDefault("sec-websocket-key")
  if key == "": return (ProtocolViolated , "sec-websocket-key header missing")
  let accept = 
    if likely(not isNil(wsserver.upgradeCallback)): wsserver.upgradeCallback()
    else: true
  if not accept:
    reply(Http400)
    suspend(200)
    closeSocket(wsserver, thesocket, CloseCalled , "ws upgrade request was not accepted: " & getRequest())
    return 
  when not defined(nimdoc):
    {.gcsafe.}:  
      let 
        sh = secureHash(key & MagicString)
        acceptKey = base64.encode(decodeBase16($sh))
    reply(Http101, ["Sec-WebSocket-Accept: " & acceptKey, "Connection: Upgrade", "Upgrade: webSocket"])
  (DontClose , "")


proc handleWsUpgradehandshake() =
  var (state , errormessage) = 
    try: replyHandshake()
    except: (Excepted , "Reply to websocket upgrade failed")
  if state != DontClose:
    if state != AlreadyClosed: closeSocket(wsserver, thesocket, state, errormessage)
    return
  if not setFlags(server, thesocket, 1):
    closeSocket(wsserver, thesocket, AlreadyClosed, "socket data disappeared")
    return
  if not isNil(wsserver.afterUpgradeCallback):
    wsserver.afterUpgradeCallback()


proc getMessage*(): string =
  return ws.request[0 ..< ws.requestlen]

{.experimental: "views".}
proc getMessageview*(ws: HttpContext): openArray[char] =
  return ws.request.toOpenArray(0, ws.requestlen - 1)


proc isMessage*(message: string): bool =
  if ws.requestlen != message.len: return false
  for i in countup(0, ws.requestlen - 1):
    if ws.request[i] != ws.request[i]: return false
  true


proc sendNonblocking(server: WebsocketServer, socket: posix.SocketHandle, text: ptr string, sent: int = 0): (SendState , int) =
  server.log(DEBUG, $getThreadId() & " writeToWebSocket " & $socket.int & ": " & text[])
  if socket == INVALID_SOCKET: return (Err , 0)
  let len = text[].len
  if sent == len: return (Delivered , 0)
  let ret = send(socket, addr text[sent], len - sent, MSG_DONTWAIT)
  if ret < 1:
    if ret == -1:
      let lastError = osLastError().int
      if lastError in [EAGAIN.int, EWOULDBLOCK.int]: return (Continue , 0)
      let cause =
        if lasterror in [2,9]: AlreadyClosed
        elif lasterror == 14: EFault
        elif lasterror == 32: ConnectionLost
        elif lasterror == 104: ClosedbyClient
        else: NetErrored
      let errormsg = osErrorMsg(OSErrorCode(lastError))
      server.closeSocket(socket, cause, " send error: " & $lastError & " " & errormsg)
      echo socket, " send error: " & $lastError & " " & errormsg
    elif ret < -1: server.closeSocket(socket, Excepted, "Send error")
    return (Err , 0)
  if sent + ret == len: return (Delivered , ret)
  return (Continue , ret)


proc closeSocketsInFlight(server: WebsocketServer, sockets: seq[posix.SocketHandle], states: seq[State]): int =
  for i in 0 ..< states.len:
    if states[i].sendstate == Continue:
      server.closeSocket(sockets[i], TimedOut, "")
      {.gcsafe.}:
        withLock(sendlock): sendingsockets.excl(sockets[i])
      result.inc


var ts {.threadvar.} : Timespec
proc getSafeMonoTime(): MonoTime {.tags: [TimeEffect].} =
  discard clock_gettime(CLOCK_MONOTONIC, ts)
  privateAccess(MonoTime)
  result = MonoTime(ticks: ts.tv_sec.int64 * 1_000_000_000 + ts.tv_nsec.int64)


let ping = "PING"
let nostatuscode = ""


proc send*(server: WebsocketServer, delivery: ptr WsDelivery, timeoutsecs = 10, sleepmillisecs = 10): int {.discardable.} =
  ## Sends message to multiple websockets at once. Uses non-blocking I/O so that slow receivers do not slow down fast receivers.
  ## | Can be called from multiple threads in parallel.
  ## | `timeoutsecs`: a timeout after which sending is given up and all sockets with messages in-flight are closed
  ## | `sleepmillisecs`: if all in-flight receivers are blocking, will suspend for (sleepmillisecs * in-flight receiver count) milliseconds
  ## Returns amount of websockets that had to be closed
  server.log(TRACE, "--starts sending websockets--")
  {.gcsafe.}:
    if delivery.message[].len == 0:
      delivery.message = addr ping
      createWsHeader(4, Opcode.Pong, server.isclient)
    elif delivery.message[].startsWith(MagicClose):
      if delivery.message[].len <= MagicClose.len:
        delivery.message = addr nostatuscode
      else:
        let statuscode = delivery.message[][MagicClose.len ..< delivery.message[].len]
        delivery.message = addr statuscode
      createWsHeader(delivery.message[].len, Opcode.Close, server.isclient)
    else:
      if server.isclient: maskmessage(delivery)
      if delivery.binary: createWsHeader(delivery.message[].len, OpCode.Binary, server.isclient)
      else: createWsHeader(delivery.message[].len, Opcode.Text, server.isclient)
  delivery.states.setLen(delivery.sockets.len)
  for i in 0 ..< delivery.states.len: delivery.states[i] = (0, NotStarted)
  var
    handled = 0
    s = -1
    blockedsockets = 0
  let
    start = getSafeMonoTime()
    timeout = initDuration(seconds = timeoutsecs)
  while true:
    if shuttingdown: return -1
    
    let elapsed = getSafeMonoTime() - start
    if elapsed > timeout:
      result += server.closeSocketsInFlight(delivery.sockets, delivery.states)
      server.log(NOTICE, "send timed out")
      return result

    s.inc
    if s == delivery.sockets.len:
      s = -1
      continue
    if delivery.states[s].sendstate in [Delivered, Err]: continue

    var ret: int
    if delivery.states[s].sendstate == NotStarted:
      {.gcsafe.}:
        withLock(sendlock):
          if sendingsockets.len == MaxParallelSendingSockets:
            server.log(INFO, "MaxParallelSendingSockets reached, sleeping for " & $(sleepmillisecs) & " ms")
            suspend(sleepmillisecs)
            continue
          if not sendingsockets.contains(delivery.sockets[s]):
            sendingsockets.incl(delivery.sockets[s])
          else: continue
      var headerstate: SendState
      (headerstate , ret) = server.sendNonblocking(delivery.sockets[s], addr wsresponseheader)
      if headerstate == Delivered: delivery.states[s].sendstate = Continue # only header delivered
      else:
        delivery.states[s].sendstate = Err
        {.gcsafe.}:
          withLock(sendlock):
            sendingsockets.excl(delivery.sockets[s])
        if headerstate == Continue: server.closeSocket(delivery.sockets[s], TimedOut)
        result.inc
        handled.inc
        server.log(TRACE, "wsockets processed: " & $handled & "/" & $delivery.sockets.len)
        if handled == delivery.sockets.len: break
      continue
      
    if shuttingdown: return -1  
    (delivery.states[s].sendstate , ret) = server.sendNonblocking(delivery.sockets[s], delivery.message, delivery.states[s].sent)
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
      {.gcsafe.}:
        withLock(sendlock): sendingsockets.excl(delivery.sockets[s])
      handled.inc
      server.log(TRACE, "wsockets processed: " & $handled & "/" & $delivery.sockets.len)
      if handled >= delivery.sockets.len: break
  server.log(TRACE, "--ends sending websockets--")


proc send*(server: WebsocketServer, sockets: seq[posix.SocketHandle], message: string, binary = false, timeoutsecs = 10, sleepmillisecs = 10): bool {.discardable.} =
  when not compiles(unsafeAddr message):
    let message = message
  when not defined(nimdoc) and not defined(gcDestructors): {.fatal: "mm:arc or mm:orc required".}
  deli.sockets = sockets
  deli.message = unsafeAddr(message)
  deli.binary = binary
  let fails = send(server, unsafeAddr deli, timeoutsecs, sleepmillisecs)
  if likely(fails < 1): return true
  server.log(NOTICE, "websocket multisend failed for " & $fails & " sockets")
  return false
  

proc send*(server: WebsocketServer, socket: posix.SocketHandle, message: string, binary = false, timeoutsecs = 2, sleepmillisecs = 10): bool {.discardable.} =
  when not compiles(unsafeAddr message):
    let message = message
  when not defined(nimdoc) and not defined(gcDestructors): {.fatal: "mm:arc or mm:orc required".}
  deli.sockets.setLen(1)
  deli.sockets[0] = socket
  deli.message = unsafeAddr(message)
  deli.binary = binary
  return send(server, unsafeAddr deli, timeoutsecs, sleepmillisecs) == 0


proc sendClose*(server: WebsocketServer, socket: posix.SocketHandle, statuscode: int16 = 1000.int16, timeoutsecs = 1, sleepmillisecs = 10): bool {.discardable.} =
  ## Sends a close frame to the client, in sync with other possible deliveries going to the same socket from other threads.
  ## | `statuscode`: Available for the client. For semantics, see `https://datatracker.ietf.org/doc/html/rfc6455#section-7.4.1 <https://datatracker.ietf.org/doc/html/rfc6455#section-7.4.1>`_ 
  ## Returns whether the sending was succesful (and if not, you may want to just call `closeOtherSocket`)
  {.gcsafe.}:
    var message = MagicClose
  let bytes = cast[array[0..1, char]](statuscode)
  message.add(bytes[1])
  message.add(bytes[0])
  deli.sockets.setLen(1)
  deli.sockets[0] = socket
  deli.message = addr message
  deli.binary = true
  return send(server, unsafeAddr deli, timeoutsecs, sleepmillisecs) == 0
 

proc handleWsRequest*() {.gcsafe, nimcall, raises: [].} =
  prepareHttpContext()
  let flags = getFlags(wsserver, thesocket)
  if unlikely(flags == -1):
    wsserver.log(DEBUG, "websocket " & $thesocket & " disappeared")
    return
  elif unlikely(flags == 0):
    if wsserver.isclient:
      if not readHeader(): closeSocket(wsserver, thesocket, NetErrored, "Websocket header read failed")
      if not setFlags(wsserver, thesocket, 1):
        closeSocket(wsserver, thesocket, NetErrored, " websocket disappeared at hanshake")
        return
    else: handleWsUpgradehandshake()
    return
  receiveWs()
  case opcode:
    of Ping:
      let pingmsg = ""
      if not wsserver.send(thesocket, pingmsg, false, 5):
        wsserver.log(NOTICE, "websocket " & $thesocket & " blocking, could not autoreply to Ping")
    of Close:
      {.gcsafe.}:
        let statuscode = getMessage()
        let message = MagicClose & statuscode
        if not wsserver.send(thesocket, message, false, 1):
          wsserver.log(NOTICE, "websocket blocking, could not autoacknowledge close")
        if statuscode == "": closeSocket(wsserver, thesocket, ClosedbyClient, "1005")
        else: closeSocket(wsserver, thesocket, ClosedbyClient, $(byte(statuscode[1]) + 256*byte(statuscode[0])))
    of WsFail: return
    else: {.gcsafe.}:
      if likely(ws.requestlen > 0 and not isNil(wsserver.messageCallback)): wsserver.messageCallback()


proc handleWsThreadInitialization*(gserver: GuildenServer) =
  handleHttpThreadInitialization(gserver)
  maskkey = newString(4)


proc initWebsocketServer*(wsserver: WebsocketServer, upgradecallback: WsUpgradeCallback, afterupgradecallback: WsAfterUpgradeCallback,
 onwsmessagecallback: WsMessageCallback, loglevel = LogLevel.WARN) =
  initHttpServer(wsserver, loglevel, true, Compact, ["sec-websocket-key"])
  wsserver.handlerCallback = handleWsRequest
  wsserver.upgradeCallback = upgradecallback
  wsserver.afterUpgradeCallback = afterupgradecallback
  wsserver.messageCallback = onwsmessagecallback
  wsserver.internalThreadInitializationCallback = handleWsThreadInitialization
 

proc newWebsocketServer(upgradecallback: WsUpgradeCallback, afterupgradecallback: WsAfterUpgradeCallback,
 onwsmessagecallback: WsMessageCallback, loglevel = LogLevel.WARN): WebsocketServer =
  result = new WebsocketServer
  initWebsocketServer(result, upgradecallback, afterupgradecallback, onwsmessagecallback, loglevel)
  
{.warning[Deprecated]:off.}
proc newWebsocketServer*(upgradecallback: WsUpgradeCallback, afterupgradecallback: WsAfterUpgradeCallback,
 onwsmessagecallback: WsMessageCallback, deprecatedOnclosesocketcallback: DeprecatedOnCloseSocketCallback, loglevel = LogLevel.WARN): WebsocketServer =
  result = newWebsocketServer(upgradecallback, afterupgradecallback, onwsmessagecallback, loglevel)
  result.deprecatedOnclosesocketcallback = deprecatedOnclosesocketcallback
{.warning[Deprecated]:on.}

proc newWebsocketServer*(upgradecallback: WsUpgradeCallback, afterupgradecallback: WsAfterUpgradeCallback,
 onwsmessagecallback: WsMessageCallback, onClosesocketcallback: OnCloseSocketCallback = nil, loglevel = LogLevel.WARN): WebsocketServer =
  result = newWebsocketServer(upgradecallback, afterupgradecallback, onwsmessagecallback, loglevel)
  result.onClosesocketcallback = onClosesocketcallback

when not defined(debug):
  {.pop.}