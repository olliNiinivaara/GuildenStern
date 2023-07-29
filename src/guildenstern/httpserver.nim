from os import osLastError, osErrorMsg, OSErrorCode
from posix import recv
from strutils import find, parseInt, isLowerAscii, toLowerAscii
import httpcore
export httpcore
import strtabs
export strtabs
import guildenserver
export guildenserver


const
  MaxHeaderLength* {.intdefine.} = 10000
  MaxRequestLength* {.intdefine.} = 100000

type
  HttpServer* = ref object of GuildenServer
    requestCallback: proc(){.gcsafe, nimcall, raises: [].}
    parserequestline: bool
    parseheaders: bool
    hascontent: bool

  HttpHandler* {.inheritable.} = ref object of GuildenHandler
    request*: string
    requestlen*: int
    uristart*: int
    urilen*: int
    methlen*: int
    bodystart*: int
    headers*: StringTableRef


proc isHttpHandler*(): bool = return guildenhandler is HttpHandler

template http*(): untyped = HttpHandler(guildenhandler)

template server*(): untyped = HttpServer(guildenhandler.socketdata.server)


proc prepareHttpHandler*(socketdata: ptr SocketData) {.inline.} =
  if unlikely(guildenhandler == nil): guildenhandler = new HttpHandler
  if unlikely(http.request.len != MaxRequestLength + 1): http.request = newString(MaxRequestLength + 1)
  http.socketdata = socketdata
  if server.parseheaders and http.headers == nil: http.headers = newStringTable()
  http.requestlen = 0
  http.uristart = 0
  http.urilen = 0
  http.methlen = 0
  http.bodystart = -1


template checkRet*() =
  if unlikely(shuttingdown): return false
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
      server.doCloseSocket(http.socketdata, cause, osErrorMsg(OSErrorCode(lastError))) 
    elif ret < -1:  server.doCloseSocket(http.socketdata, Excepted, getCurrentExceptionMsg())
    else: server.doCloseSocket(http.socketdata, ClosedbyClient, "") 
    return false
  
  if http.methlen == 0:
    if unlikely(http.requestlen + ret < 13):
      server.log(WARN, "too short request (" & $ret & "): " & http.request)
      server.doCloseSocket(http.socketdata, ProtocolViolated, "")
      return false
    while http.methlen < ret and http.request[http.methlen] != ' ': http.methlen.inc
    if unlikely(http.methlen == ret):
      server.log(WARN, "http method missing")
      server.doCloseSocket(http.socketdata, ProtocolViolated, "")
      return false
    if unlikely(http.request[0 .. 1] notin ["GE", "PO", "HE", "PU", "DE", "CO", "OP", "TR", "PA"]):
      server.log(WARN, "invalid http method: " & http.request[0 .. 12])
      server.doCloseSocket(http.socketdata, ProtocolViolated, "")
      return false


proc parseRequestLine*(): bool {.gcsafe, raises: [].} =
  var i = http.methlen + 1
  let start = i
  while i < http.requestlen and http.request[i] != ' ': i.inc()
  http.uristart = start
  http.urilen = i - start

  if unlikely(http.requestlen < http.uristart + http.urilen + 9):
    server.log(WARN, "parseRequestLine: no version")
    (server.doCloseSocket(http.socketdata, ProtocolViolated, ""); return false)

  if unlikely(http.request[http.uristart + http.urilen + 1] != 'H' or http.request[http.uristart + http.urilen + 8] != '1'):
    server.log(WARN, "request not HTTP/1.1: " & http.request[http.uristart + http.urilen + 1 .. http.uristart + http.urilen + 8])
    (server.doCloseSocket(http.socketdata, ProtocolViolated, ""); return false)
  server.log(DEBUG, $server.port & "/" & $http.socketdata.socket &  ": " & http.request[0 .. http.uristart + http.urilen + 8])
  true


proc isHeaderreceived*(previouslen, currentlen: int): bool =
  if currentlen < 4: return false
  if http.request[currentlen-4] == '\c' and http.request[currentlen-3] == '\l' and http.request[currentlen-2] == '\c' and
  http.request[currentlen-1] == '\l':
    http.bodystart = currentlen
    return true

  var i = if previouslen > 4: previouslen - 4 else: previouslen
  while i <= currentlen - 4:
    if http.request[i] == '\c' and http.request[i+1] == '\l' and http.request[i+2] == '\c' and http.request[i+3] == '\l':
      http.bodystart = i + 4
      return true
    inc i
  false


proc getContentLength*(): int {.raises: [].} =
  const length  = "content-length: ".len
  var start = http.request.find("content-length: ")
  if start == -1: start = http.request.find("Content-Length: ")
  if start == -1: return 0
  var i = start + length
  while i < http.requestlen and http.request[i] != '\c': i += 1
  if i == http.requestlen: return 0
  try: return parseInt(http.request[start + length ..< i])
  except CatchableError:
    server.log(WARN, "could not parse content-length from: " & http.request)
    return 0
  
 
proc getUri*(): string {.raises: [].} =
  if http.urilen == 0: return
  return http.request[http.uristart ..< http.uristart + http.urilen]


proc isUri*(uri: string): bool {.raises: [].} =
  if http.urilen != uri.len: return false
  for i in 0 ..< http.urilen:
    if http.request[http.uristart + i] != uri[i]: return false
  return true


proc startsUri*(uristart: string): bool {.raises: [].} =
  if http.urilen < uristart.len: return false
  for i in 0 ..< uristart.len:
    if http.request[http.uristart + i] != uristart[i]: return false
  true


proc getMethod*(): string {.raises: [].} =
  if http.methlen == 0: return
  return http.request[0 ..< http.methlen]


proc isMethod*(amethod: string): bool {.raises: [].} =
  if http.methlen != amethod.len: return false
  for i in 0 ..< http.methlen:
    if http.request[i] != amethod[i]: return false
  true


proc getHeaders*(): string =
  if http.bodystart < 1: return http.request
  http.request[0 .. http.bodystart - 4]


proc getBodystart*(): int {.inline.} =
  http.bodystart


proc getBodylen*(): int =
  if http.bodystart < 1: return 0
  return http.requestlen - http.bodystart


when compiles((var x = 1; var vx: var int = x)):
  # --experimental:views is enabled
  proc getBody(): openArray[char] =
    if http.bodystart < 1: return http.request.toOpenArray(0, -1)
    else: return http.request.toOpenArray(ctx.bodystart, ctx.requestlen - 1)
else:
  proc getBody*(): string =
    if http.bodystart < 1: return ""
    return http.request[http.bodystart ..< http.requestlen]


proc isBody*(body: string): bool =
  let len = http.requestlen - http.bodystart
  if  len != body.len: return false
  for i in http.bodystart ..< http.bodystart + len:
    if http.request[i] != body[i]: return false
  true


proc getRequest*(): string =
  return http.request[0 ..< http.requestlen]


proc getMessage*(): string =
  return http.request[0 ..< http.requestlen]


proc isRequest*(request: string): bool =
  if http.requestlen != http.request.len: return false
  for i in countup(0, http.requestlen - 1):
    if http.request[i] != http.request[i]: return false
  true


proc isHeader*(headerfield: string, value: string): bool =
  assert(server.parseheaders)
  try: return http.headers[headerfield] == value
  except: return false


proc parseHeaders*(fields: openArray[string], toarray: var openArray[string]) =
  assert(fields.len == toarray.len)
  for j in 0 ..< fields.len: assert(fields[j][0].isLowerAscii(), "Header field names must be given in all lowercase, wrt. " & fields[j])
  var value = false
  var current: (string, string) = ("", "")
  var found = 0
  var i = 0

  while i <= http.requestlen - 4:
    case http.request[i]
    of '\c':
      if http.request[i+1] == '\l' and http.request[i+2] == '\c' and http.request[i+3] == '\l':
        let index = fields.find(current[0])
        if index != -1: toarray[index] = current[1]
        return
    of ':':
      if value: current[1].add(':')
      value = true
    of ' ':
      if value:
        if current[1].len != 0: current[1].add(http.request[i])
      else: current[0].add(http.request[i])
    of '\l':
      let index = fields.find(current[0])
      if index != -1:
        toarray[index] = current[1]
        found += 1
        if found == toarray.len: return
      value = false
      current = ("", "")
    else:
      if value: current[1].add(http.request[i])
      else: current[0].add((http.request[i]).toLowerAscii())
    i.inc


proc parseHeaders*(headers: StringTableRef) =
  # note: does not clear table first
  var value = false
  var current: (string, string) = ("", "")
  var i = 0
  while i <= http.requestlen - 4:
    case http.request[i]
    of '\c':
      if http.request[i+1] == '\l' and http.request[i+2] == '\c' and http.request[i+3] == '\l':
        headers[current[0]] = current[1]
        return
    of ':':
      if value: current[1].add(':')
      value = true
    of ' ':
      if value:
        if current[1].len != 0: current[1].add(http.request[i])
      else: current[0].add(http.request[i])
    of '\l':
      headers[current[0]] = current[1]
      value = false
      current = ("", "")
    else:
      if value: current[1].add(http.request[i])
      else: current[0].add(http.request[i].toLowerAscii())
    i.inc


proc receiveAllHttp(): bool {.gcsafe, raises:[] .} =
  var expectedlength = MaxRequestLength + 1
  while true:
    if shuttingdown: return false
    let ret = recv(http.socketdata.socket, addr http.request[http.requestlen], expectedlength - http.requestlen, 0)
    checkRet()
    let previouslen = http.requestlen
    http.requestlen += ret

    if http.requestlen >= MaxRequestLength:
      server.doCloseSocket(http.socketdata, ProtocolViolated, "recvHttp: Max request size exceeded")
      return false

    if http.requestlen == expectedlength: break

    if not isHeaderreceived(previouslen, http.requestlen):
      if http.requestlen >= MaxHeaderLength:
        server.doCloseSocket(http.socketdata,ProtocolViolated, "recvHttp: Max header size exceeded" )
        return false
      continue

    let contentlength = getContentLength()
    if contentlength == 0: return true
    expectedlength = http.bodystart + contentlength
    if http.requestlen == expectedlength: break
  server.log(DEBUG, $server.port & "/" & $http.socketdata.socket & ": " & http.request[http.bodystart .. http.bodystart + http.requestlen - 1])
  true


proc receiveHeader*(): bool {.gcsafe, raises:[].} =
  while true:
    if shuttingdown: return false
    let ret = recv(http.socketdata.socket, addr http.request[http.requestlen], 1 + MaxHeaderLength - http.requestlen, 0)
    checkRet()
    http.requestlen += ret
    if http.requestlen > MaxHeaderLength:
      server.doCloseSocket(guildenhandler.socketdata, ProtocolViolated, "receiveHeader: Max header size exceeded")
      return false
    if http.request[http.requestlen-4] == '\c' and http.request[http.requestlen-3] == '\l' and
     http.request[http.requestlen-2] == '\c' and http.request[http.requestlen-1] == '\l': break
  return http.requestlen > 0


proc handleRequest(data: ptr SocketData) {.gcsafe, nimcall, raises: [].} =
  prepareHttpHandler(data)
  if not server.hascontent:
    if not receiveHeader(): return
  else:
    if not receiveAllHttp(): return
  if server.parserequestline and not parseRequestLine(): return
  if server.parseHeaders: parseHeaders(http.headers)

  {.gcsafe.}: server.requestCallback()


proc newHttpServer*(onrequestcallback: proc(){.gcsafe, nimcall, raises: [].}, parserequestline = true, parseheaders = true, hascontent = true): HttpServer =
  ## Initializes the headerctx handler for given port with given request callback.
  ## By setting global `parserequestline` to false all HeaderCtxs become pass-through handlers
  ## that do no handling for the request.
  result = new HttpServer
  result.registerHandler(handleRequest)
  result.requestCallback = onrequestcallback
  result.parserequestline = parserequestline
  result.parseheaders = parseheaders
  result.hascontent = hascontent


include httpresponse


template reply*(code: HttpCode, headers: openArray[string]) =
  reply(code, nil, headers)

template reply*(body: string) =
  when compiles(unsafeAddr body):
    reply(Http200, unsafeAddr body, nil)
  else: {.fatal: "posix.send requires taking pointer to body, but body has no address".}

template reply*(code: HttpCode, body: string) =
  when compiles(unsafeAddr body):
    reply(code, unsafeAddr body, nil)
  else: {.fatal: "posix.send requires taking pointer to body, but body has no address".} 

template reply*(code: HttpCode, body: string, headers: openArray[string]) =
  when compiles(unsafeAddr body):
    reply(code, unsafeAddr body, headers)
  else: {.fatal: "posix.send requires taking pointer to body, but body has no address".}

template replyStart*(code: HttpCode, contentlength: int, firstpart: string, headers: openArray[string]): bool =
  when compiles(unsafeAddr firstpart):
    replyStart(code, contentlength, unsafeAddr firstpart, headers)
  else: {.fatal: "posix.send requires taking pointer to firstpart, but firstpart has no address".}

template replyStart*(contentlength: int, firstpart: string): bool =
  when compiles(unsafeAddr firstpart):
    replyStart(Http200, contentlength, unsafeAddr firstpart, nil)
  else: {.fatal: "posix.send requires taking pointer to firstpart, but firstpart has no address".}

template replyMore*(bodypart: string): bool =
  when compiles(unsafeAddr bodypart):
    replyMore(unsafeAddr bodypart)
  else: {.fatal: "posix.send requires taking pointer to bodypart, but bodypart has no address".}

template replyLast*(lastpart: string) =
  when compiles(unsafeAddr lastpart):
    replyLast(unsafeAddr lastpart)
  else: {.fatal: "posix.send requires taking pointer to lastpart, but lastpart has no address".} 

template replyLast*() =
  replyLast(nil)
