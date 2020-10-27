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


proc writeVersion*(gs: ptr GuildenServer, fd: posix.SocketHandle): bool {.inline, gcsafe, raises: [].} =
  var ret: int
  try:
    {.gcsafe.}: ret = send(fd, unsafeAddr version[0], 9, intermediateflags)
  except:
    let lastError = osLastError().int
    echo "writeVersion except: " & $lastError & " " & osErrorMsg(OSErrorCode(lastError)) 
    gs.closeFd(fd)
    return false
  if ret != 9:
    echo "writeVersion väärä ret: " & $ret 
    gs.closeFd(fd)
    return false
  echo "write onnistu"
  true


proc writeCode*(gs: ptr GuildenServer, fd: posix.SocketHandle, code: HttpCode) {.inline, gcsafe, raises: [].} =
  var ret: int
  try:
    if code == Http200:
      {.gcsafe.}: ret = send(fd, unsafeAddr http200string[0], 6, intermediateflags)
    else:
      let codestring = $code
      ret = send(fd, unsafeAddr codestring[0], codestring.len, 0)
  except:
    let lastError = osLastError().int
    echo "writecode except: " & $lastError & " " & osErrorMsg(OSErrorCode(lastError)) 
    gs.closeFd(fd)


proc sendHttp*(gs: ptr GuildenServer, fd: posix.SocketHandle, text: ptr string, length: int = -1, flags: cint = intermediateflags) {.inline, gcsafe, raises: [].} =
  var length = length
  if length == -1: length = text[].len
  if length == 0: return
  var bytessent = 0
  var retries = 0
  while bytessent < length:
    let ret =
      try: send(fd, unsafeAddr text[bytessent], length - bytessent, flags)
      except: -1111
    echo "lähti: ", ret
    echo text[]
    if ret < 0:
      let lastError = osLastError().int
      echo "sendHttp: " & $lastError & " " & osErrorMsg(OSErrorCode(lastError))
      gs.closeFd(fd)
      return
    if gs.serverstate == Shuttingdown: return
    if ret == 0:
      retries.inc(10)
      sleep(retries)
    retries.inc
    if retries > 100:
      gs.closeFd(fd)
      return
    bytessent.inc(ret)

let
  shortdivider = "\c\L"
  longdivider = "\c\L\c\L"
  contentlength = "Content-Length: "
  zerocontent = "Content-Length: 0\c\L"


proc replyCode*(ctx: Ctx, code: HttpCode) {.inline, gcsafe, raises: [].} =
  if not writeVersion(ctx.gs, ctx.socket): return
  writeCode(ctx.gs, ctx.socket, code)
  {.gcsafe.}: sendHttp(ctx.gs, ctx.socket, unsafeAddr longdivider, longdivider.len, lastflag)


proc reply*(ctx: Ctx, code: HttpCode, body: ptr string, headers: ptr string) {.inline, gcsafe, raises: [].} =
  {.gcsafe.}: 
    if not writeVersion(ctx.gs, ctx.socket):
      ctx.currentexceptionmsg = "http out: " & osErrorMsg(OSErrorCode(osLastError().int))
      return

    writeCode(ctx.gs, ctx.socket, code)
    sendHttp(ctx.gs, ctx.socket, unsafeAddr shortdivider)
    
    if body == nil and headers == nil:
      sendHttp(ctx.gs, ctx.socket, unsafeAddr zerocontent)
      sendHttp(ctx.gs, ctx.socket, unsafeAddr longdivider, longdivider.len, lastflag)
      return
    
    if headers != nil and headers[].len > 0:
      sendHttp(ctx.gs, ctx.socket, headers)
      sendHttp(ctx.gs, ctx.socket, unsafeAddr shortdivider)

    if code == Http303 or code == Http304:      
      sendHttp(ctx.gs, ctx.socket, unsafeAddr shortdivider, shortdivider.len, lastflag)
      # closeFd(ctx.gs, ctx.socket) should close or not?
      return

    if body == nil or body[].len == 0:      
      sendHttp(ctx.gs, ctx.socket, unsafeAddr zerocontent)
      sendHttp(ctx.gs, ctx.socket, unsafeAddr longdivider, longdivider.len, lastflag)
      return

    sendHttp(ctx.gs, ctx.socket, unsafeAddr contentlength)    
    let bodylen = $body[].len
    sendHttp(ctx.gs, ctx.socket, unsafeAddr bodylen)
    sendHttp(ctx.gs, ctx.socket, unsafeAddr longdivider)
    sendHttp(ctx.gs, ctx.socket, body, -1, lastflag)
     

proc doReplyHeaders*(ctx: Ctx, code: HttpCode=Http200, headers: ptr string) {.inline, gcsafe, raises: [].} =
  reply(ctx, code, nil, headers)


proc replyHeaders*(ctx: Ctx, headers: openArray[string], code: HttpCode=Http200) {.inline.} =
  let joinedheaders = headers.join("\c\L")
  reply(ctx, code, nil, unsafeAddr joinedheaders)
  

#[proc reply*(ctx: Ctx, code: HttpCode, body: string, headers: openArray[string]) {.inline.} =
  discard doReply(ctx.gs, ctx.socket, code,  body, headers.join("\c\L"))

proc reply*(ctx: Ctx, code: HttpCode, body: string, headers: seq[string]) {.inline.} =
  discard doReply(ctx.gs, ctx.socket, code,  body, headers.join("\c\L"))

proc joinHeaders(headers: openArray[seq[string]]): string {.inline.} =
  for x in headers.low .. headers.high:
    for y in headers[x].low .. headers[x].high:
      if y > 0 or x > 0: result.add("\c\L")
      result.add(headers[x][y])

proc reply*(ctx: Ctx, code: HttpCode, body: string,  headers: openArray[seq[string]]) {.inline.} =
  discard doReply(ctx.gs, ctx.socket, code,  body, joinHeaders(headers))

proc reply*(ctx: Ctx, body: string, code=Http200) {.inline.} =
  discard doReply(ctx.gs, ctx.socket, code, body)

proc replyHeaders*(ctx: Ctx, headers: openArray[string], code: HttpCode=Http200) {.inline.} =
  discard doReplyHeaders(ctx.gs, ctx.socket, code, headers.join("\c\L"))

proc replyHeaders*(ctx: Ctx, headers: seq[string], code: HttpCode=Http200) {.inline.} =
  discard doReplyHeaders(ctx.gs, ctx.socket, code, headers.join("\c\L"))

proc replyHeaders*(ctx: Ctx, headers: openArray[seq[string]], code: HttpCode=Http200) {.inline.} =
  discard doReplyHeaders(ctx.gs, ctx.socket, code, joinHeaders(headers))

proc reply*(ctx: Ctx, headers: openArray[string]) {.inline.} =
  discard doReply(ctx, Http200, headers.join("\c\L"))

proc reply*(ctx: Ctx, headers: seq[string]) {.inline.} =
  discard doreply(ctx, Http200, headers.join("\c\L"))

proc reply*(ctx: Ctx, headers: openArray[seq[string]]) {.inline.} =
  discard doreply(ctx, Http200, joinHeaders(headers))]#

{.pop.}