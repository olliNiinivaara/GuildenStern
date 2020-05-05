import nativesockets, streams, net, posix, os, strutils, std/sha1, base64
import guildenserver
from httpcore import Http400
from httpout import writeToHttp, replyCode


const OpCodeText = 0x1

{.push checks: off.}

proc writeWs(c: GuildenVars, toSocket = NullHandle.SocketHandle, text: StringStream = nil): bool =
  let fd =
    if toSocket != NullHandle.SocketHandle: toSocket
    else: c.fd
  let wslen = c.wsheader.getPosition()
  if wslen > 0:
    try:
      let ret = send(fd, addr c.wsheader.data[0], wslen, 0)
      if ret != wslen: return false
    except:
      c.currentexceptionmsg = "websocket write: " & getCurrentExceptionMsg()
      return false

  let stream = if text != nil: text else: c.sendbuffer
  let len = stream.getPosition()
  const maxSize = 1024*1024
  var sent = 0
  while sent < len:
    let datalen = min(len - sent, maxSize)
    var bytesSent = 0
    var trials = 0
    var ret: int      
    while bytesSent < datalen:
      try:
        ret = send(fd, addr stream.data[sent], datalen, 0)
      except:
        c.currentexceptionmsg = "websocket write: " & getCurrentExceptionMsg()
        return false
      if ret == -1:
        c.currentexceptionmsg = "websocket write: " & osErrorMsg(osLastError())
        return false
      if ret == 0:
        trials = trials + 1
        if trials < 5:
           sleep(100 + 100*trials)
           echo "backoff triggered"
           continue
           #TODO: real backoff strategy
        return false
      bytesSent.inc(ret)
    sent.inc(datalen)
  return true


proc createWsHeader(c: GuildenVars, len: int) =
  c.wsheader.setPosition(0)
  var b0 = (OpcodeText.uint8 and 0x0f) # 0th byte: opcodes and flags
  b0 = b0 or 128u8 # 1st bit set indicates that this is the final fragment in a message.

  # Payload length can be 7 bits, 7+16 bits, or 7+64 bits.
  # 1st byte: playload len start and mask bit.
  var b1 = 0u8
  if len <= 125: b1 = len.uint8
  elif len > 125 and len <= 0xffff: b1 = 126u8
  else: b1 = 127u8

  c.wsheader.write(b0)
  c.wsheader.write(b1)
    
  # Only need more bytes if data len is 7+16 bits, or 7+64 bits.
  if len > 125 and len <= 0xffff:
    c.wsheader.write(nativesockets.htons(len.uint16))
  elif len > 0xffff:
    c.wsheader.write char((len shr 56) and 255)
    c.wsheader.write char((len shr 48) and 255)
    c.wsheader.write char((len shr 40) and 255)
    c.wsheader.write char((len shr 32) and 255)
    c.wsheader.write char((len shr 24) and 255)
    c.wsheader.write char((len shr 16) and 255)
    c.wsheader.write char((len shr 8) and 255)
    c.wsheader.write char(len and 255)


proc writeToWs*(c: GuildenVars, toSocket = NullHandle.SocketHandle, text: StringStream = nil): bool =
  if text == nil:
    c.createWsHeader(c.sendbuffer.getPosition())
    result = c.writeWs(toSocket, c.sendbuffer)
  else:
    c.createWsHeader(text.getPosition())
    result =  c.writeWs(toSocket, text)
  if not result:
    try: discard posix.close(c.fd)
    except: discard


proc nibbleFromChar(c: char): int =
  ## Converts hex chars like `0` to 0 and `F` to 15.
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

{.pop.}

proc wsHandshake*(c: GuildenVars): bool =
  try:
    var keystart = c.recvbuffer.data.find("\lSec-Websocket-Key: ")
    if keystart == -1: keystart = c.recvbuffer.data.find("\lsec-websocket-key: ")
    if keystart == -1: keystart = c.recvbuffer.data.find("\lSec-WebSocket-Key: ")
    if keystart == -1:
      c.currentexceptionmsg = "websocket handshake: Sec-Websocket-Key header not found"  
      return false
    let 
      key = c.recvbuffer.data.substr(keystart+20, keystart+43)
      sh = secureHash(key & "258EAFA5-E914-47DA-95CA-C5AB0DC85B11")
      acceptKey = base64.encode(decodeBase16($sh))
    
    var responce = "HTTP/1.1 101 Web Socket Protocol Handshake\c\L" # responce [Sic]
    responce.add("Sec-WebSocket-Accept: " & acceptKey & "\c\L")
    responce.add("Connection: Upgrade\c\L")
    responce.add("Upgrade: webSocket\c\L")

    responce.add "\c\L"
    c.currentexceptionmsg = c.fd.writeToHttp(responce)
    return c.currentexceptionmsg == ""
  except:
    c.replyCode(Http400)
    c.currentexceptionmsg = "websocket handshake: " & getCurrentExceptionMsg()
    return false