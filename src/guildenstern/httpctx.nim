from streams import StringStream, newStringStream, getPosition, setPosition, write
from strutils import find #toLowerAscii, parseInt
import strtabs
import guildenserver, httpout
export httpout

const
  MaxRequestLength* {.intdefine.} = 1000
  MaxResponseLength* {.intdefine.} = 100000

type
  HttpCtx* = ref object of Ctx
    requestlen*: int
    path*: int
    pathlen*: int
    methlen*: int
    headers* : StringTableRef
    bodystart*: int


template newHttpCtxData*() =
  ctxpool[i].recvdata = newStringStream()
  ctxpool[i].recvdata.data.setLen(MaxRequestLength)
  ctxpool[i].senddata = newStringStream()
  ctxpool[i].senddata.data.setLen(MaxResponseLength)
  ctxpool[i].headers = newStringTable(modeCaseInsensitive)


template initHttpCtx*() =
  result.currentexceptionmsg.setLen(0)
  result.requestlen = 0
  result.path = 0
  result.pathlen = 0
  result.methlen = 0
  result.bodystart = 0
  result.headers.clear()
  try:
    result.senddata.setPosition(0)
    result.recvdata.setPosition(0)
  except: (echo "Nim internal error"; return)


proc handleError*(ctx: HttpCtx): bool {.inline.} =
  if ctx.currentexceptionmsg == "": return false
  if ctx.gs.errorHandler != nil: ctx.gs.errorHandler(ctx)
  else:
    if defined(fulldebug): echo ctx.currentexceptionmsg
  true


proc parseRequestLine*(ctx: HttpCtx) {.gcsafe, raises: [].} =
  try: ctx.requestlen = ctx.recvdata.getPosition()-1
  except: return
  if ctx.requestlen < 14: return
  while ctx.methlen < ctx.requestlen and ctx.recvdata.data[ctx.methlen] != ' ': ctx.methlen.inc
  if ctx.methlen == ctx.requestlen: return
  var i = ctx.methlen + 1
  let start = i
  while i < ctx.requestlen and ctx.recvdata.data[i] != ' ': i.inc()
  ctx.path = start
  ctx.pathlen = i - start
  

proc getPath*(ctx: HttpCtx): string {.raises: [].} =
  if ctx.pathlen == 0: return
  return ctx.recvdata.data[ctx.path ..< ctx.path + ctx.pathlen]


proc isPath*(ctx: HttpCtx, apath: string): bool {.raises: [].} =
  if ctx.pathlen != apath.len: return false
  for i in 0 ..< ctx.pathlen:
    if ctx.recvdata.data[ctx.path + i] != apath[i]: return false
  return true


proc pathStarts*(ctx: HttpCtx, pathstart: string): bool {.raises: [].} =
  if ctx.pathlen < pathstart.len: return false
  for i in 0 ..< pathstart.len:
    if ctx.recvdata.data[ctx.path + i] != pathstart[i]: return false
  true


proc getMethod*(ctx: HttpCtx): string {.raises: [].} =
  if ctx.methlen == 0: return
  return ctx.recvdata.data[0 ..< ctx.methlen]


proc isMethod*(ctx: HttpCtx, amethod: string): bool {.raises: [].} =
  if ctx.methlen != amethod.len: return false
  for i in 0 ..< ctx.methlen:
    if ctx.recvdata.data[i] != amethod[i]: return false
  true


proc getBody*(ctx: HttpCtx): string =
  ctx.recvdata.data[ctx.bodystart ..< ctx.requestlen]
  

proc isBody*(ctx: HttpCtx, body: string): bool {.raises: [].} =
  let len = ctx.requestlen - ctx.bodystart
  if  len != body.len: return false
  for i in ctx.bodystart ..< ctx.bodystart + len:
    if ctx.recvdata.data[i] != body[i]: return false
  true


proc getHeader*(ctx: HttpCtx, field: string): string {.raises: [].} =
  assert(ctx.headers.len == 0, "ctx.headers is already parsed and available")
  let start = ctx.recvdata.data.find(field) + field.len + 1
  if start == field.len + 1 or start >= ctx.requestlen: return ""
  var i = start + 1
  while i < ctx.requestlen and ctx.recvdata.data[i] != '\l': i += 1
  if i == ctx.requestlen: return ""
  return ctx.recvdata.data[start .. i]


proc parseHeaders*(ctx: HttpCtx) =
  var value = false
  var current: (string, string) = ("", "")
  var i = 0

  while i <= ctx.requestlen - 4:
    case ctx.recvdata.data[i]
    of '\c':
      if ctx.recvdata.data[i+1] == '\l' and ctx.recvdata.data[i+2] == '\c' and ctx.recvdata.data[i+3] == '\l':
        if ctx.requestlen > i + 4: ctx.bodystart = i + 4
        return
    of ':':
      if value: current[1].add(':')
      value = true
    of ' ':
      if value:
        if current[1].len != 0: current[1].add(ctx.recvdata.data[i])
      else: current[0].add(ctx.recvdata.data[i])
    of '\l':
      echo current
      ctx.headers[current[0]] = current[1]
      value = false
      current = ("", "")
    else:
      if value: current[1].add(ctx.recvdata.data[i])
      else: current[0].add(ctx.recvdata.data[i])
    i.inc


proc writeData*(ctx: HttpCtx, str: ptr string): bool {.raises: [].} =
  try: 
    if ctx.senddata.getPosition() + str[].len() > MaxResponseLength: return false
    ctx.senddata.write(str[])
  except:  return false
  true


proc sendData*(ctx: HttpCtx, code: HttpCode=Http200, headers: ptr string) =
  let length = ctx.senddata.getPosition()
  if length == 0: reply(ctx, code, nil, headers)
  else: reply(ctx, code, addr ctx.senddata.data, headers)