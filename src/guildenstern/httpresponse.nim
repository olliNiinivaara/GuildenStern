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


proc writeVersion*(ctx: HttpCtx): bool {.inline, gcsafe, raises: [].} =
  var ret: int
  {.gcsafe.}: ret = send(ctx.socketdata.socket, unsafeAddr version[0], 9, intermediateflags)
  checkRet(ctx)
  if ret != 9:
    ctx.closeSocket(ProtocolViolated)
    return false
  true


proc writeCode*(ctx: HttpCtx, code: HttpCode): bool {.inline, gcsafe, raises: [].} =
  var ret: int
  try:
    if code == Http200:
      {.gcsafe.}: ret = send(ctx.socketdata.socket, unsafeAddr http200string[0], 6, intermediateflags)
    elif code == Http204:
      {.gcsafe.}: ret = send(ctx.socketdata.socket, unsafeAddr http204string[0], 14, 0)
    else:
      let codestring = $code # slow...
      ret = send(ctx.socketdata.socket, unsafeAddr codestring[0], codestring.len.cint, 0)
  except:
    ret = -2
  checkRet(ctx)
  ctx.gs[].log(DEBUG, "writeCode " & $ctx.socketdata.socket & ": " & $code)
  true


proc writeToSocket*(ctx: HttpCtx, text: ptr string, length: int, flags = intermediateflags): bool {.inline, gcsafe, raises: [].} =
  if length == 0: return true
  var bytessent = 0
  var backoff = 1
  while bytessent < length:
    let ret =
      try: send(ctx.socketdata.socket, unsafeAddr text[bytessent], (length - bytessent).cint, flags + DONTWAIT)
      except: -2
    if ret == -1 and osLastError().cint in [EAGAIN, EWOULDBLOCK]:
      sleep(backoff)
      backoff *= 2
      if backoff > 3000:
        ctx.closeSocket(TimedOut, "didn't write to socket")
        return false
      continue
    checkRet(ctx)
    bytessent.inc(ret)
  if text[0] != '\c':
    ctx.gs[].log(DEBUG, "writeToSocket " & $ctx.socketdata.socket & ": " & text[0 ..< length])
  true


let
  shortdivider = "\c\L"
  longdivider = "\c\L\c\L"
  contentlen = "Content-Length: "
  zerocontent = "Content-Length: 0\c\L"


proc replyCode(ctx: HttpCtx, code: HttpCode = Http200) {.inline, gcsafe, raises: [].} =
  if not writeVersion(ctx): return
  if not writeCode(ctx, code): return
  {.gcsafe.}:
      if code == Http204: discard writeToSocket(ctx, unsafeAddr longdivider, longdivider.len, lastflag)
      else:
        if not writeToSocket(ctx, unsafeAddr shortdivider, shortdivider.len): return
        if not writeToSocket(ctx, unsafeAddr zerocontent, zerocontent.len): return
        discard writeToSocket(ctx, unsafeAddr shortdivider, shortdivider.len, lastflag)
      

proc reply*(ctx: HttpCtx, code: HttpCode, body: ptr string, lengths: string, length: int, headers: ptr string, moretocome: bool): bool {.gcsafe, raises: [].} =
  if body == nil and headers == nil: (ctx.replyCode(code); return true)
  let finalflag = if moretocome: intermediateflags else: lastflag
  {.gcsafe.}: 
    if not writeVersion(ctx): return false 
    if not writeCode(ctx, code): return false
    if not writeToSocket(ctx, unsafeAddr shortdivider, shortdivider.len): return false

    if headers != nil and headers[].len > 0:
      if not writeToSocket(ctx, headers, headers[].len): return false
      if not writeToSocket(ctx, unsafeAddr shortdivider, shortdivider.len): return false

    if code == Http101 or code == Http304:      
      return writeToSocket(ctx, unsafeAddr shortdivider, shortdivider.len, finalflag)
      
    if length < 1:      
      if not writeToSocket(ctx, unsafeAddr zerocontent, zerocontent.len): return false
      return writeToSocket(ctx, unsafeAddr longdivider, longdivider.len, finalflag)
      
    if not writeToSocket(ctx, unsafeAddr contentlen, contentlen.len): return false
    if not writeToSocket(ctx, unsafeAddr lengths, lengths.len): return false
    if not writeToSocket(ctx, unsafeAddr longdivider, longdivider.len): return false
    return writeToSocket(ctx, body, length, finalflag)


proc replyStart*(ctx: HttpCtx, code: HttpCode, contentlength: int, firstpart: ptr string, headers: ptr string = nil): bool {.inline, gcsafe, raises: [].} =
  {.gcsafe.}: 
    if not writeVersion(ctx): return false 
    if not writeCode(ctx, code): return false
    if not writeToSocket(ctx, unsafeAddr shortdivider, shortdivider.len): return false

    if headers != nil and headers[].len > 0:
      if not writeToSocket(ctx, headers, headers[].len): return false
      if not writeToSocket(ctx, unsafeAddr shortdivider, shortdivider.len): return false
      
    if not writeToSocket(ctx, unsafeAddr contentlen, contentlen.len): return false
    let length = $contentlength
    if not writeToSocket(ctx, unsafeAddr length, length.len): return false
    if not writeToSocket(ctx, unsafeAddr longdivider, longdivider.len): return false
    return writeToSocket(ctx, firstpart, firstpart[].len)


proc reply*(ctx: HttpCtx, code: HttpCode, body: ptr string = nil, headers: ptr string = nil) {.inline, gcsafe, raises: [].} =
  let length = if body == nil: 0 else: body[].len
  if likely(ctx.reply(code, body, $length, length, headers, false)): ctx.gs[].log(TRACE, "reply ok")
  else: ctx.gs[].log(INFO, $ctx.socketdata.socket & ": reply failed")


proc reply*(ctx: HttpCtx, code: HttpCode, body: ptr string, headers: openArray[string]) {.inline, gcsafe, raises: [].} =
  let joinedheaders = headers.join("\c\L")
  reply(ctx, code, body, unsafeAddr joinedheaders)


proc replyStart*(ctx: HttpCtx, code: HttpCode, contentlength: int, body: ptr string, headers: openArray[string]): bool {.inline, gcsafe, raises: [].} =
  let joinedheaders = headers.join("\c\L")
  replyStart(ctx, code, contentlength, body, unsafeAddr joinedheaders)


proc replyMore*(ctx: HttpCtx, bodypart: ptr string, partlength: int = -1): bool {.inline, gcsafe, raises: [].} =
  let length = if partlength != -1: partlength else: bodypart[].len
  return ctx.writeToSocket(bodypart, length)


proc replyLast*(ctx: HttpCtx, lastpart: ptr string, partlength: int = -1) {.inline, gcsafe, raises: [].} =
  let length = if partlength != -1: partlength else: lastpart[].len
  discard ctx.writeToSocket(lastpart, length, lastflag)

{.pop.}
