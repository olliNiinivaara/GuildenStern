from os import osLastError, osErrorMsg, OSErrorCode
from posix import recv
from strutils import find, parseInt, isLowerAscii, toLowerAscii
from httpcore import Http200, Http204
export Http200, Http204
import strtabs
import guildenserver


const
  MaxHeaderLength* {.intdefine.} = 10000
  MaxRequestLength* {.intdefine.} = 100000

type
  HttpCtx* = ref object of Ctx
    requestlen*: int
    uristart*: int
    urilen*: int
    methlen*: int
    bodystart*: int

var
  request* {.threadvar.}: string

proc initHttpCtx*(ctx: HttpCtx, gs: ptr GuildenServer, socketdata: ptr SocketData) {.inline.} =
  if request.len < MaxRequestLength + 1: request = newString(MaxRequestLength + 1)
  ctx.gs = gs
  ctx.socketdata = socketdata
  ctx.requestlen = 0
  ctx.uristart = 0
  ctx.urilen = 0
  ctx.methlen = 0
  ctx.bodystart = -1


template checkRet*() =
  if shuttingdown: return false
  if ret < 1:
    if ret == -1:
      let lastError = osLastError().int
      let cause =
        # https://www-numi.fnal.gov/offline_software/srt_public_context/WebDocs/Errors/unix_system_errors.html
        if lasterror in [2,9]: AlreadyClosed
        elif lasterror == 11: TimedOut
        elif lasterror == 32: ConnectionLost
        elif lasterror == 104: ClosedbyClient
        else: NetErrored
      ctx.closeSocket(cause, osErrorMsg(OSErrorCode(lastError)))
    elif ret < -1: ctx.closeSocket(Excepted, getCurrentExceptionMsg())
    else: ctx.closeSocket(ClosedbyClient)        
    return false
  
  if ctx.methlen > 0: return true

  if ctx.requestlen + ret < 13:
    when defined(fulldebug): echo "too short request (", ret,"): ", request
    (ctx.closeSocket(ProtocolViolated); return false)

  while ctx.methlen < ret and request[ctx.methlen] != ' ': ctx.methlen.inc
  if ctx.methlen == ret:
    when defined(fulldebug): echo "http method missing"
    (ctx.closeSocket(ProtocolViolated); return false)
  if request[0 .. 1] notin ["GE", "PO", "HE", "PU", "DE", "CO", "OP", "TR", "PA"]:
    when defined(fulldebug): echo "invalid http method: ", request[0 .. 12]
    (ctx.closeSocket(ProtocolViolated); return false)


proc parseRequestLine*(ctx: HttpCtx): bool {.gcsafe, raises: [].} =
  var i = ctx.methlen + 1
  let start = i
  while i < ctx.requestlen and request[i] != ' ': i.inc()
  ctx.uristart = start
  ctx.urilen = i - start
  if ctx.requestlen < ctx.uristart + ctx.urilen + 9:
    when defined(fulldebug): echo ("parseRequestLine: no version")
    (ctx.closeSocket(ProtocolViolated); return false)
  
  if request[ctx.uristart + ctx.urilen + 1] != 'H' or request[ctx.uristart + ctx.urilen + 8] != '1':
    when defined(fulldebug): echo "request not HTTP/1.1: ", request[ctx.uristart + ctx.urilen + 1 .. ctx.uristart + ctx.urilen + 8]
    (ctx.closeSocket(ProtocolViolated); return false)
  when defined(fulldebug): echo ctx.socketdata.port, ": ", request[0 .. ctx.uristart + ctx.urilen + 8]
  true


proc isHeaderreceived*(ctx: HttpCtx, previouslen, currentlen: int): bool =
  if currentlen < 4: return false
  if request[currentlen-4] == '\c' and request[currentlen-3] == '\l' and request[currentlen-2] == '\c' and
  request[currentlen-1] == '\l':
    ctx.bodystart = currentlen
    return true

  var i = if previouslen > 4: previouslen - 4 else: previouslen
  while i <= currentlen - 4:
    if request[i] == '\c' and request[i+1] == '\l' and request[i+2] == '\c' and request[i+3] == '\l':
      ctx.bodystart = i + 4
      return true
    inc i
  false


proc getContentLength*(ctx: HttpCtx): int {.raises: [].} =
  const length  = "content-length: ".len
  var start = request.find("content-length: ")
  if start == -1: start = request.find("Content-Length: ")
  if start == -1: return 0
  var i = start + length
  while i < ctx.requestlen and request[i] != '\c': i += 1
  if i == ctx.requestlen: return 0
  try: return parseInt(request[start + length ..< i])
  except:
    when defined(fulldebug): echo "could not parse content-length from: ", request[start + length ..< i], "; ", getCurrentExceptionMsg()
    return 0
  
 
proc getUri*(ctx: HttpCtx): string {.raises: [].} =
  if ctx.urilen == 0: return
  return request[ctx.uristart ..< ctx.uristart + ctx.urilen]


proc isUri*(ctx: HttpCtx, uri: string): bool {.raises: [].} =
  if ctx.urilen != uri.len: return false
  for i in 0 ..< ctx.urilen:
    if request[ctx.uristart + i] != uri[i]: return false
  return true


proc startsUri*(ctx: HttpCtx, uristart: string): bool {.raises: [].} =
  if ctx.urilen < uristart.len: return false
  for i in 0 ..< uristart.len:
    if request[ctx.uristart + i] != uristart[i]: return false
  true


proc getMethod*(ctx: HttpCtx): string {.raises: [].} =
  if ctx.methlen == 0: return
  return request[0 ..< ctx.methlen]


proc isMethod*(ctx: HttpCtx, amethod: string): bool {.raises: [].} =
  if ctx.methlen != amethod.len: return false
  for i in 0 ..< ctx.methlen:
    if request[i] != amethod[i]: return false
  true


proc getHeaders*(ctx: HttpCtx): string =
  if ctx.bodystart < 1: return request
  request[0 .. ctx.bodystart - 4]


proc getBodystart*(ctx: HttpCtx): int {.inline.} =
  ctx.bodystart


proc getBodylen*(ctx: HttpCtx): int =
  if ctx.bodystart < 1: return 0
  return ctx.requestlen - ctx.bodystart


proc getBody*(ctx: HttpCtx): string =
  if ctx.bodystart < 1: return ""
  request[ctx.bodystart ..< ctx.requestlen]
  

proc isBody*(ctx: HttpCtx, body: string): bool =
  let len = ctx.requestlen - ctx.bodystart
  if  len != body.len: return false
  for i in ctx.bodystart ..< ctx.bodystart + len:
    if request[i] != body[i]: return false
  true


proc getRequest*(ctx: HttpCtx): string =
  request[0 ..< ctx.requestlen]


proc parseHeaders*(ctx: HttpCtx, fields: openArray[string], toarray: var openArray[string]) =
  assert(fields.len == toarray.len)
  for j in 0 ..< fields.len: assert(fields[j][0].isLowerAscii(), "Header field names must be given in all lowercase, wrt. " & fields[j])
  var value = false
  var current: (string, string) = ("", "")
  var found = 0
  var i = 0

  while i <= ctx.requestlen - 4:
    case request[i]
    of '\c':
      if request[i+1] == '\l' and request[i+2] == '\c' and request[i+3] == '\l':
        let index = fields.find(current[0])
        if index != -1: toarray[index] = current[1]
        return
    of ':':
      if value: current[1].add(':')
      value = true
    of ' ':
      if value:
        if current[1].len != 0: current[1].add(request[i])
      else: current[0].add(request[i])
    of '\l':
      let index = fields.find(current[0])
      if index != -1:
        toarray[index] = current[1]
        found += 1
        if found == toarray.len: return
      value = false
      current = ("", "")
    else:
      if value: current[1].add(request[i])
      else: current[0].add((request[i]).toLowerAscii())
    i.inc


proc parseHeaders*(ctx: HttpCtx, headers: StringTableRef) =
  # note: does not clear table first
  var value = false
  var current: (string, string) = ("", "")
  var i = 0
  while i <= ctx.requestlen - 4:
    case request[i]
    of '\c':
      if request[i+1] == '\l' and request[i+2] == '\c' and request[i+3] == '\l':
        headers[current[0]] = current[1]
        return
    of ':':
      if value: current[1].add(':')
      value = true
    of ' ':
      if value:
        if current[1].len != 0: current[1].add(request[i])
      else: current[0].add(request[i])
    of '\l':
      headers[current[0]] = current[1]
      value = false
      current = ("", "")
    else:
      if value: current[1].add(request[i])
      else: current[0].add(request[i].toLowerAscii())
    i.inc


include httpresponse


template reply*(ctx: HttpCtx, code: HttpCode, headers: openArray[string]) =
  reply(ctx, code, nil, headers)

template reply*(ctx: HttpCtx, body: string) =
  when compiles(unsafeAddr body):
    reply(ctx, Http200, unsafeAddr body, nil)
  else: {.fatal: "posix.send requires taking pointer to body, but body has no address".}

template reply*(ctx: HttpCtx,  code: HttpCode, body: string) =
  when compiles(unsafeAddr body):
    reply(ctx, code, unsafeAddr body, nil)
  else: {.fatal: "posix.send requires taking pointer to body, but body has no address".} 

template reply*(ctx: HttpCtx, code: HttpCode, body: string, headers: openArray[string]) =
  when compiles(unsafeAddr body):
    reply(ctx, code, unsafeAddr body, headers)
  else: {.fatal: "posix.send requires taking pointer to body, but body has no address".}

template replyStart*(ctx: HttpCtx, code: HttpCode, contentlength: int, firstpart: string, headers: openArray[string]): bool =
  when compiles(unsafeAddr firstpart):
    replyStart(ctx, code, contentlength, unsafeAddr firstpart, headers)
  else: {.fatal: "posix.send requires taking pointer to firstpart, but firstpart has no address".}

template replyStart*(ctx: HttpCtx, code: HttpCode, contentlength: int, firstpart: string): bool =
  when compiles(unsafeAddr firstpart):
    replyStart(ctx, code, contentlength, unsafeAddr firstpart, nil)
  else: {.fatal: "posix.send requires taking pointer to firstpart, but firstpart has no address".}

template replyMore*(ctx: HttpCtx, bodypart: string): bool =
  when compiles(unsafeAddr bodypart):
    replyMore(ctx, unsafeAddr bodypart)
  else: {.fatal: "posix.send requires taking pointer to bodypart, but bodypart has no address".}

template replyLast*(ctx: HttpCtx, lastpart: string) =
  when compiles(unsafeAddr lastpart):
    replyLast(ctx, unsafeAddr lastpart)
  else: {.fatal: "posix.send requires taking pointer to lastpart, but lastpart has no address".} 

template replyLast*(ctx: HttpCtx) =
  replyLast(ctx, nil)
