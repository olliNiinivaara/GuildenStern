import posix, net, nativesockets, os, httpcore, streams
import ../guildenserver
from strutils import join
export HttpCode, Http200

{.push checks: off.}

const
  intermediateflags = MSG_NOSIGNAL + 0x8000 # MSG_MORE
  lastflag = MSG_NOSIGNAL

let
  emptyreply =  "HTTP/1.1 204 No Content \c\L\c\L"
  version = "HTTP/1.1 "
  http200string = "200 OK"


proc replyEmpty*(h: Handler) {.inline, gcsafe, raises: [].} =
  var ret: int
  try:
    {.gcsafe.}: ret = send(h.socket, unsafeAddr emptyreply[0], 28, lastflag)
  except:
    h.gs.closeFd(h.socket)
    return
  if ret != 28: h.gs.closeFd(h.socket)


proc writeVersion*(gs: GuildenServer, fd: posix.SocketHandle): bool {.inline, gcsafe, raises: [].} =
  var ret: int
  try:
    {.gcsafe.}: ret = send(fd, unsafeAddr version, 9, intermediateflags)
  except:
    gs.closeFd(fd)
    return false
  if ret != 9:
    gs.closeFd(fd)
    return false
  true


proc writeCode*(gs: GuildenServer, fd: posix.SocketHandle, code: HttpCode) {.inline, gcsafe, raises: [].} =
  var ret: int
  try:
    if code == Http200:
      {.gcsafe.}: ret = send(fd, unsafeAddr http200string, 6, intermediateflags)
    else:
      let codestring = $code
      ret = send(fd, unsafeAddr codestring, 3, 0)
  except: gs.closeFd(fd)
  if ret != 6: gs.closeFd(fd) # brutal...


proc sendHttp*(gs: GuildenServer, fd: posix.SocketHandle, text: ptr string, length: int = -1, flags: cint = intermediateflags) {.inline, gcsafe, raises: [].} =
  var length = length
  if length == -1: length = text[].len
  if length == 0: return
  var bytessent = 0
  var retries = 0
  while bytessent < length:
    let ret =
      try: send(fd, unsafeAddr text[bytessent], length - bytessent, flags)
      except: -1
    if ret == -1:
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


proc doReply*(h: Handler, code: HttpCode, body: ptr string, headers: ptr string) {.inline, gcsafe, raises: [].} =
  {.gcsafe.}: 
    if not writeVersion(h.gs, h.socket):
      h.currentexceptionmsg = "http out: " & osErrorMsg(OSErrorCode(osLastError().int))
      return
    writeCode(h.gs, h.socket, code)
    sendHttp(h.gs, h.socket, unsafeAddr shortdivider)
    if body == nil or body[].len == 0:  sendHttp(h.gs, h.socket, unsafeAddr zerocontent)
    else:
      sendHttp(h.gs, h.socket, unsafeAddr contentlength)
      let bodylen = $body[].len
      sendHttp(h.gs, h.socket, unsafeAddr bodylen)
    if headers != nil and headers[].len > 0:
      sendHttp(h.gs, h.socket,  unsafeAddr shortdivider)
      sendHttp(h.gs, h.socket, headers)
    if body == nil or body[].len == 0:      
      sendHttp(h.gs, h.socket, unsafeAddr longdivider, 8, lastflag)
    else:
      sendHttp(h.gs, h.socket, unsafeAddr longdivider)
      sendHttp(h.gs, h.socket, body, -1, lastflag)


proc doReplyHeaders*(h: Handler, code: HttpCode=Http200, headers: ptr string) {.inline, gcsafe, raises: [].} =
  doReply(h, code, nil, headers)
  

#[proc reply*(h: Handler, code: HttpCode, body: string, headers: openArray[string]) {.inline.} =
  discard doReply(h.gs, h.socket, code,  body, headers.join("\c\L"))

proc reply*(h: Handler, code: HttpCode, body: string, headers: seq[string]) {.inline.} =
  discard doReply(h.gs, h.socket, code,  body, headers.join("\c\L"))

proc joinHeaders(headers: openArray[seq[string]]): string {.inline.} =
  for x in headers.low .. headers.high:
    for y in headers[x].low .. headers[x].high:
      if y > 0 or x > 0: result.add("\c\L")
      result.add(headers[x][y])

proc reply*(h: Handler, code: HttpCode, body: string,  headers: openArray[seq[string]]) {.inline.} =
  discard doReply(h.gs, h.socket, code,  body, joinHeaders(headers))

proc reply*(h: Handler, body: string, code=Http200) {.inline.} =
  discard doReply(h.gs, h.socket, code, body)

proc replyHeaders*(h: Handler, headers: openArray[string], code: HttpCode=Http200) {.inline.} =
  discard doReplyHeaders(h.gs, h.socket, code, headers.join("\c\L"))

proc replyHeaders*(h: Handler, headers: seq[string], code: HttpCode=Http200) {.inline.} =
  discard doReplyHeaders(h.gs, h.socket, code, headers.join("\c\L"))

proc replyHeaders*(h: Handler, headers: openArray[seq[string]], code: HttpCode=Http200) {.inline.} =
  discard doReplyHeaders(h.gs, h.socket, code, joinHeaders(headers))

proc reply*(h: Handler, headers: openArray[string]) {.inline.} =
  discard doReply(h, Http200, headers.join("\c\L"))

proc reply*(h: Handler, headers: seq[string]) {.inline.} =
  discard doreply(h, Http200, headers.join("\c\L"))

proc reply*(h: Handler, headers: openArray[seq[string]]) {.inline.} =
  discard doreply(h, Http200, joinHeaders(headers))]#

proc replyCode*(h: Handler, code: HttpCode) {.inline, gcsafe, raises: [].} =
  doReply(h, code, nil, nil)

{.pop.}