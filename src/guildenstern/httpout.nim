import posix, net, nativesockets, os, httpcore, streams
import guildenserver
from strutils import join
export HttpCode, Http200

{.push checks: off.}

const
  intermediateflags = MSG_NOSIGNAL + 0x8000 # MSG_MORE
  lastflag = MSG_NOSIGNAL

let
  version = "HTTP/1.1 "
  http200string = "200 OK"
  http204string = "204 No Content"


proc writeVersion*(ctx: HttpCtx): bool {.inline, gcsafe, raises: [].} =
  var ret: int
  {.gcsafe.}: ret = send(ctx.socketdata.socket, unsafeAddr version[0], 9, intermediateflags)
  checkRet()
  if ret != 9:
    ctx.closeSocket()
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
      ret = send(ctx.socketdata.socket, unsafeAddr codestring[0], codestring.len, 0)
  except:
    ret = -2
  checkRet()
  true

proc sendToSocket*(ctx: HttpCtx, text: ptr string, length: int = -1, flags = intermediateflags): bool {.inline, gcsafe, raises: [].} =
  var length = length
  if length == -1: length = text[].len
  if length == 0: return true
  var bytessent = 0
  while bytessent < length:
    let ret =
      try: send(ctx.socketdata.socket, unsafeAddr text[bytessent], length - bytessent, flags)
      except: -2
    checkRet()
    if ctx.gs.serverstate == Shuttingdown: return false
    bytessent.inc(ret)
  true


let
  shortdivider = "\c\L"
  longdivider = "\c\L\c\L"
  contentlength = "Content-Length: "
  zerocontent = "Content-Length: 0\c\L"


proc replyCode*(ctx: HttpCtx, code: HttpCode) {.inline, gcsafe, raises: [].} =
  if not writeVersion(ctx): return
  if not writeCode(ctx, code): return
  {.gcsafe.}:
    if code != Http204: discard sendToSocket(ctx, unsafeAddr zerocontent)
    discard sendToSocket(ctx, unsafeAddr longdivider, longdivider.len, lastflag)


proc replyStart*(ctx: HttpCtx, code: HttpCode, body: ptr string = nil, headers: ptr string = nil, length = -1, moretocome = true): bool {.gcsafe, raises: [].} =
  let finalflag = if moretocome: intermediateflags else: lastflag
  if body == nil and headers == nil: (ctx.replyCode(code); return false)
  {.gcsafe.}: 
    if not writeVersion(ctx): return false 
    if not writeCode(ctx, code): return false
    if not sendToSocket(ctx, unsafeAddr shortdivider): return false

    if headers != nil and headers[].len > 0:
      if not sendToSocket(ctx, headers): return false
      if not sendToSocket(ctx, unsafeAddr shortdivider): return false

    if code == Http303 or code == Http304:      
      return sendToSocket(ctx, unsafeAddr shortdivider, shortdivider.len, finalflag)
      # closeSocket(ctx.gs, ctx.socketdata.socket) probably not?
      
    if body == nil or body[].len == 0 or length == 0:      
      if not sendToSocket(ctx, unsafeAddr zerocontent): return false
      return sendToSocket(ctx, unsafeAddr longdivider, longdivider.len, finalflag)
      
    if not sendToSocket(ctx, unsafeAddr contentlength): return false
    let len = if length == -1: $body[].len else: $length
    if not sendToSocket(ctx, unsafeAddr len): return false
    if not sendToSocket(ctx, unsafeAddr longdivider): return false
    return sendToSocket(ctx, body, -1, finalflag)


proc reply*(ctx: HttpCtx, code: HttpCode, body: ptr string = nil, headers: ptr string = nil, length = -1) {.inline, gcsafe, raises: [].} =
  discard ctx.replyStart(code, body, headers, length, false)


proc writeToSocketBuffered*(ctx: HttpCtx, text: ptr string = nil, length = -1): bool {.gcsafe, raises: [].} =
  return ctx.sendToSocket(text, -1)


proc writeToSocket*(ctx: HttpCtx, text: ptr string = nil, length = -1): bool {.gcsafe, raises: [].} =
  return ctx.sendToSocket(text, -1, lastflag)
     

proc doReplyHeaders*(ctx: HttpCtx, code: HttpCode=Http200, headers: ptr string) {.inline, gcsafe, raises: [].} =
  reply(ctx, code, nil, headers)


proc replyHeaders*(ctx: HttpCtx, headers: openArray[string], code: HttpCode=Http200) {.inline.} =
  let joinedheaders = headers.join("\c\L")
  reply(ctx, code, nil, unsafeAddr joinedheaders)
  

#[proc reply*(ctx: Ctx, code: HttpCode, body: string, headers: openArray[string]) {.inline.} =
  discard doReply(ctx.gs, ctx.socketdata.socket, code,  body, headers.join("\c\L"))

proc reply*(ctx: Ctx, code: HttpCode, body: string, headers: seq[string]) {.inline.} =
  discard doReply(ctx.gs, ctx.socketdata.socket, code,  body, headers.join("\c\L"))

proc joinHeaders(headers: openArray[seq[string]]): string {.inline.} =
  for x in headers.low .. headers.high:
    for y in headers[x].low .. headers[x].high:
      if y > 0 or x > 0: result.add("\c\L")
      result.add(headers[x][y])

proc reply*(ctx: Ctx, code: HttpCode, body: string,  headers: openArray[seq[string]]) {.inline.} =
  discard doReply(ctx.gs, ctx.socketdata.socket, code,  body, joinHeaders(headers))

proc reply*(ctx: Ctx, body: string, code=Http200) {.inline.} =
  discard doReply(ctx.gs, ctx.socketdata.socket, code, body)

proc replyHeaders*(ctx: Ctx, headers: openArray[string], code: HttpCode=Http200) {.inline.} =
  discard doReplyHeaders(ctx.gs, ctx.socketdata.socket, code, headers.join("\c\L"))

proc replyHeaders*(ctx: Ctx, headers: seq[string], code: HttpCode=Http200) {.inline.} =
  discard doReplyHeaders(ctx.gs, ctx.socketdata.socket, code, headers.join("\c\L"))

proc replyHeaders*(ctx: Ctx, headers: openArray[seq[string]], code: HttpCode=Http200) {.inline.} =
  discard doReplyHeaders(ctx.gs, ctx.socketdata.socket, code, joinHeaders(headers))

proc reply*(ctx: Ctx, headers: openArray[string]) {.inline.} =
  discard doReply(ctx, Http200, headers.join("\c\L"))

proc reply*(ctx: Ctx, headers: seq[string]) {.inline.} =
  discard doreply(ctx, Http200, headers.join("\c\L"))

proc reply*(ctx: Ctx, headers: openArray[seq[string]]) {.inline.} =
  discard doreply(ctx, Http200, joinHeaders(headers))]#

{.pop.}