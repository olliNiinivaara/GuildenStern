{.push hints: off.}
import posix, net, nativesockets, os, httpcore
import guildenserver
from strutils import join
{.pop.}

{.push checks: off.}

const
  DONTWAIT = 0x40.cint
  intermediateflags = MSG_NOSIGNAL + 0x8000 # MSG_MORE
  lastflag = MSG_NOSIGNAL

let
  version = "HTTP/1.1 "
  http200string = "200 OK"
  http204string = "204 No Content"


proc writeVersion*(): bool {.inline, gcsafe, raises: [].} =
  var ret: int
  {.gcsafe.}: ret = send(http.socketdata.socket, unsafeAddr version[0], 9, intermediateflags)
  checkRet()
  if ret != 9:
    server.doCloseSocket(http.socketdata, ProtocolViolated, "")
    return false
  true


proc writeCode*(code: HttpCode): bool {.inline, gcsafe, raises: [].} =
  var ret: int
  try:
    if code == Http200:
      {.gcsafe.}: ret = send(http.socketdata.socket, unsafeAddr http200string[0], 6, intermediateflags)
    elif code == Http204:
      {.gcsafe.}: ret = send(http.socketdata.socket, unsafeAddr http204string[0], 14, 0)
    else:
      let codestring = $code # slow...
      ret = send(http.socketdata.socket, unsafeAddr codestring[0], codestring.len.cint, 0)
  except CatchableError:
    ret = -2
  checkRet()
  server.log(DEBUG, "writeCode " & $http.socketdata.socket & ": " & $code)
  true


proc writeToSocket*(text: ptr string, length: int, flags = intermediateflags): bool {.inline, gcsafe, raises: [].} =
  if length == 0: return true
  var bytessent = 0
  var backoff = 1
  while bytessent < length:
    let ret =
      try: send(http.socketdata.socket, unsafeAddr text[bytessent], (length - bytessent).cint, flags + DONTWAIT)
      except CatchableError: -2
    if ret == -1 and osLastError().cint in [EAGAIN, EWOULDBLOCK]:
      sleep(backoff)
      backoff *= 2
      if backoff > 3000:
        server.doCloseSocket(http.socketdata, TimedOut, "didn't write to socket")
        return false
      continue
    checkRet()
    bytessent.inc(ret)
  if text[0] != '\c':
    server.log(DEBUG, "writeToSocket " & $http.socketdata.socket & ": " & text[0 ..< length])
  true


let
  shortdivider = "\c\L"
  longdivider = "\c\L\c\L"
  contentlen = "Content-Length: "
  zerocontent = "Content-Length: 0\c\L"


proc replyCode*(code: HttpCode = Http200) {.inline, gcsafe, raises: [].} =
  if not writeVersion(): return
  if not writeCode(code): return
  {.gcsafe.}:
      if code == Http204: discard writeToSocket(unsafeAddr longdivider, longdivider.len, lastflag)
      else:
        if not writeToSocket(unsafeAddr shortdivider, shortdivider.len): return
        if not writeToSocket(unsafeAddr zerocontent, zerocontent.len): return
        discard writeToSocket(unsafeAddr shortdivider, shortdivider.len, lastflag)
      

proc reply*(code: HttpCode, body: ptr string, lengths: string, length: int, headers: ptr string, moretocome: bool): bool {.gcsafe, raises: [].} =
  if body == nil and headers == nil: (replyCode(code); return true)
  let finalflag = if moretocome: intermediateflags else: lastflag
  {.gcsafe.}: 
    if not writeVersion(): return false 
    if not writeCode(code): return false
    if not writeToSocket(unsafeAddr shortdivider, shortdivider.len): return false

    if headers != nil and headers[].len > 0:
      if not writeToSocket(headers, headers[].len): return false
      if not writeToSocket(unsafeAddr shortdivider, shortdivider.len): return false

    if code == Http101 or code == Http304:      
      return writeToSocket(unsafeAddr shortdivider, shortdivider.len, finalflag)
      
    if length < 1:      
      if not writeToSocket(unsafeAddr zerocontent, zerocontent.len): return false
      return writeToSocket(unsafeAddr longdivider, longdivider.len, finalflag)
      
    if not writeToSocket(unsafeAddr contentlen, contentlen.len): return false
    if not writeToSocket(unsafeAddr lengths, lengths.len): return false
    if not writeToSocket(unsafeAddr longdivider, longdivider.len): return false
    return writeToSocket(body, length, finalflag)


proc replyStart*(code: HttpCode, contentlength: int, firstpart: ptr string, headers: ptr string = nil): bool {.inline, gcsafe, raises: [].} =
  {.gcsafe.}: 
    if not writeVersion(): return false 
    if not writeCode(code): return false
    if not writeToSocket(unsafeAddr shortdivider, shortdivider.len): return false

    if headers != nil and headers[].len > 0:
      if not writeToSocket(headers, headers[].len): return false
      if not writeToSocket(unsafeAddr shortdivider, shortdivider.len): return false
      
    if not writeToSocket(unsafeAddr contentlen, contentlen.len): return false
    let length = $contentlength
    if not writeToSocket(unsafeAddr length, length.len): return false
    if not writeToSocket(unsafeAddr longdivider, longdivider.len): return false
    return writeToSocket(firstpart, firstpart[].len)


proc reply*(code: HttpCode, body: ptr string = nil, headers: ptr string = nil) {.inline, gcsafe, raises: [].} =
  let length = if body == nil: 0 else: body[].len
  if likely(reply(code, body, $length, length, headers, false)): server.log(TRACE, "reply ok")
  else: server.log(INFO, $http.socketdata.socket & ": reply failed")


proc reply*(code: HttpCode, body: ptr string, headers: openArray[string]) {.inline, gcsafe, raises: [].} =
  let joinedheaders = headers.join("\c\L")
  reply(code, body, unsafeAddr joinedheaders)


proc replyStart*(code: HttpCode, contentlength: int, body: ptr string, headers: openArray[string]): bool {.inline, gcsafe, raises: [].} =
  let joinedheaders = headers.join("\c\L")
  replyStart(code, contentlength, body, unsafeAddr joinedheaders)


proc replyMore*(bodypart: ptr string, partlength: int = -1): bool {.inline, gcsafe, raises: [].} =
  let length = if partlength != -1: partlength else: bodypart[].len
  return writeToSocket(bodypart, length)


proc replyLast*(lastpart: ptr string, partlength: int = -1) {.inline, gcsafe, raises: [].} =
  let length = if partlength != -1: partlength else: lastpart[].len
  discard writeToSocket(lastpart, length, lastflag)

{.pop.}
