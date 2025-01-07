from std/strutils import find, parseInt, isLowerAscii, toLowerAscii
from posix import MSG_PEEK


proc parseMethod*(): bool =
  if unlikely(http.requestlen < 13):
    server.log(WARN, "too short request: " & http.request)
    closeSocket(server, thesocket, ProtocolViolated, "")
    return false
  while http.methlen < http.requestlen and http.request[http.methlen] != ' ': http.methlen.inc
  if unlikely(http.methlen == http.requestlen):
    server.log(WARN, "http method missing")
    closeSocket(server, thesocket, ProtocolViolated, "")
    return false
  if unlikely(http.request[0 .. 1] notin ["GE", "PO", "HE", "PU", "DE", "CO", "OP", "TR", "PA"]):
    server.log(WARN, "invalid http method: " & http.request[0 .. 12])
    closeSocket(server, thesocket, ProtocolViolated, "")
    return false
  return true
  

proc parseRequestLine*(): bool {.gcsafe, raises: [].} =
  if not parseMethod(): return false
  var i = http.methlen + 1
  let start = i
  while i < http.requestlen and http.request[i] != ' ': i.inc()
  http.uristart = start
  http.urilen = i - start

  if unlikely(http.requestlen < http.uristart + http.urilen + 9):
    server.log(WARN, "parseRequestLine: no version")
    closeSocket(server, thesocket, ProtocolViolated, "")
    return false

  if unlikely(http.request[http.uristart + http.urilen + 1] != 'H' or http.request[http.uristart + http.urilen + 8] != '1'):
    server.log(WARN, "request not HTTP/1.1: " & http.request[http.uristart + http.urilen + 1 .. http.uristart + http.urilen + 8])
    closeSocket(server, thesocket, ProtocolViolated, "")
    return false
  server.log(DEBUG, $server.port & "/" & $thesocket &  ": " & http.request[0 .. http.uristart + http.urilen + 8])
  true


proc getContentLength*(): bool {.raises: [].} =
  const length  = "content-length: ".len
  var start = http.request.find("content-length: ")
  if start == -1: start = http.request.find("Content-Length: ")
  if start == -1: return true
  var i = start + length
  while i < http.requestlen and http.request[i] != '\c': i += 1
  if i == http.requestlen: return true
  try:
    http.contentlength = parseInt(http.request[start + length ..< i])
    return true
  except:
    closeSocket(server, thesocket, ProtocolViolated, "could not parse content-length")
    return false


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
  
 
proc getUri*(): string {.raises: [].} =
  ## Returns the uri as a string copy
  doAssert(server.parserequestline == true)
  if http.urilen == 0: return
  return http.request[http.uristart ..< http.uristart + http.urilen]


proc isUri*(uri: string): bool {.raises: [].} =
  ## Compares the uri without making a string copy
  assert(server.parserequestline)
  if http.urilen != uri.len: return false
  for i in 0 ..< http.urilen:
    if http.request[http.uristart + i] != uri[i]: return false
  return true


proc startsUri*(uristart: string): bool {.raises: [].} =
  ## Compares the beginning of the uri without making a string copy
  assert(server.parserequestline)
  if http.urilen < uristart.len: return false
  for i in 0 ..< uristart.len:
    if http.request[http.uristart + i] != uristart[i]: return false
  true


proc getMethod*(): string {.raises: [].} =
  ## Returns the method as a string copy
  assert(server.parserequestline)
  if http.methlen == 0: return
  return http.request[0 ..< http.methlen]


proc isMethod*(amethod: string): bool {.raises: [].} =
  ## Compares method without making a string copy
  assert(server.parserequestline)
  if http.methlen != amethod.len: return false
  for i in 0 ..< http.methlen:
    if http.request[i] != amethod[i]: return false
  true


proc getBodylen*(): int =
  if http.bodystart < 1: return 0
  return http.requestlen - http.bodystart


when compiles((var x = 1; var vx: var int = x)):
  ## Returns the body without making a string copy.
  proc getBodyview*(http: HttpContext): openArray[char] =
    assert(server.contenttype == Compact)
    if http.bodystart < 1: return http.request.toOpenArray(0, -1)
    else: return http.request.toOpenArray(http.bodystart, http.requestlen - 1)


proc getBody*(): string =
  ## Returns the body as a string copy.  When --experimental:views compiler switch is used, there is also getBodyview proc that does not take a copy.
  if unlikely(server.contenttype != Compact):
    server.log(ERROR, "getBody is available only when server.contenttype == Compact")
    return
  if http.bodystart < 1: return ""
  return http.request[http.bodystart ..< http.requestlen]


proc isBody*(body: string): bool =
  ## Compares the body without making a string copy
  if unlikely(server.contenttype != Compact):
    server.log(ERROR, "isBody is available only when server.contenttype == Compact")
    return
  let len = http.requestlen - http.bodystart
  if  len != body.len: return false
  for i in http.bodystart ..< http.bodystart + len:
    if http.request[i] != body[i]: return false
  true


proc getRequest*(): string =
  assert(server.contenttype == Compact)
  return http.request[0 ..< http.requestlen]


proc receiveHeader(): bool {.gcsafe, raises:[].} =
  var backoff = 4
  var totalbackoff = 0
  while true:
    if shuttingdown: return false
    let ret = recv(thesocket, addr http.request[http.requestlen], 1 + server.maxheaderlength - http.requestlen, MSG_DONTWAIT)
    let state = checkSocketState(ret)
    if state == Fail: return false
    if state == SocketState.TryAgain:
      suspend(backoff)
      totalbackoff += backoff
      if totalbackoff > server.sockettimeoutms:
        if http.requestlen == 0: closeSocket(server, thesocket, TimedOut, "client sent nothing")
        else: closeSocket(server, thesocket, TimedOut, "didn't receive whole header in time")
        return false
      backoff *= 2
      continue
    totalbackoff = 0
    http.requestlen += ret
    if isHeaderreceived(http.requestlen - ret, http.requestlen): break
    if http.requestlen > server.maxheaderlength:
      closeSocket(server, thesocket, ProtocolViolated, "maximum allowed header size exceeded")
      return false
  http.contentreceived = http.requestlen - http.bodystart
  true


proc parseHeaders() =
  var value = false
  var current: (string, string) = ("", "")
  var found = 0
  var i = 0
  while i <= http.requestlen - 4:
    case http.request[i]
    of '\c':
      if http.request[i+1] == '\l' and http.request[i+2] == '\c' and http.request[i+3] == '\l':
        if http.headers.contains(current[0]): http.headers[current[0]] = current[1]
        return
    of ':':
      if value: current[1].add(':')
      value = true
    of ' ':
      if value:
        if current[1].len != 0: current[1].add(http.request[i])
      else: current[0].add(http.request[i])
    of '\l':
      if http.headers.contains(current[0]):
        http.headers[current[0]] = current[1]
        found += 1
        if found == http.headers.len: return
      value = false
      current = ("", "")
    else:
      if value: current[1].add(http.request[i])
      else: current[0].add((http.request[i]).toLowerAscii())
    i.inc


proc hasData(): bool =
  var r = recv(thesocket, addr http.probebuffer[0], 1, MSG_PEEK or MSG_DONTWAIT)
  if likely(r == 1): return true
  suspend(100)
  r = recv(thesocket, addr http.probebuffer[0], 1, MSG_PEEK or MSG_DONTWAIT)
  if likely(r == 1): return true
  closeSocket(server, thesocket, ClosedbyClient, "client sent nothing")
  return false
  #[var loops: int
  while true:
    let ret = recv(thesocket, addr http.probebuffer[0], 1, MSG_DONTWAIT)
    if ret > 0:
      echo "tuli merkki: ", http.probebuffer
      return true
    if unlikely(ret == 0):
      echo "nolla tuli"
      suspend(100)
      loops += 1
      if loops > 5:
        echo "ei taida tulla dataa"
        return false
      continue
    let lastError = osLastError().int
    if lasterror == EAGAIN.int: return false
    else: return true]#


proc readHeader*(): bool {.gcsafe, raises:[].} =
  if not hasData(): return false
  if not receiveHeader(): return false
  if server.headerfields.len == 0:
    if server.contenttype == NoBody: return true
    return getContentLength()
  parseHeaders()
  if server.contenttype != NoBody:
    try:
      if http.headers["content-length"].len > 0:
        http.contentlength = http.headers["content-length"].parseInt()
    except:
      closeSocket(server, thesocket, ProtocolViolated, "non-parseable content-length")
      return false  
  true


iterator receiveStream*(): (SocketState , string) {.gcsafe, raises: [].} =
  ## Receives a http request in chunks, yielding the state of operation and a possibly received new chuck on every iteration.
  ## With this, you can receive data incrementally without worries about main memory usage.
  ## See examples/streamingposttest.nim for a concrete working example of how to use this iterator.
  #let theserver = getServer()
  if http.contentlength == 0: yield (Complete , "")
  else:
    if http.contentreceived == http.contentlength:
      if server.contenttype == Streaming: yield (Progress , http.request[http.bodystart ..< http.bodystart + http.contentlength])
      yield (Complete , "")
    else:
      if server.contenttype == Streaming: yield (Progress , http.request[http.bodystart ..< http.bodystart + http.contentreceived])
      var continues = true
      while continues:
        if shuttingdown:
          yield (Fail , "")
          continues = false
        else:
          let recvsize =
            if http.contentlength - http.contentreceived > server.bufferlength: server.bufferlength
            else: http.contentlength - http.contentreceived
          let position =
            if server.contenttype == Streaming: 0
            else: http.bodystart + http.contentreceived
          let ret: int64 = recv(thesocket, addr http.request[position], recvsize, MSG_DONTWAIT)
          let state = checkSocketState(ret)
          if ret > 0: http.contentreceived += ret
          http.requestlen =
            if server.contenttype == Streaming: ret
            else: http.bodystart + http.contentreceived
          if state == Fail:
            yield (Fail , "")
            continues = false
          elif state == Complete or http.contentlength == http.contentreceived:
            if server.contenttype == Streaming: yield(Progress , http.request[0 ..< ret])
            yield(Complete , "")
            continues = false
          elif state == TryAgain:
            yield (TryAgain , "")
          else:
            if server.contenttype == Streaming:
              yield(Progress , http.request[0 ..< ret])
            else: yield(Progress , "")


proc receiveToSingleBuffer(): bool =
  var backoff = 4
  var totalbackoff = 0
  for (state , chunk) in receiveStream():
    case state:
      of TryAgain:
        suspend(backoff)
        totalbackoff += backoff
        if totalbackoff > server.sockettimeoutms:
          closeSocket(server, thesocket, TimedOut, "didn't read all contents from socket")
          return false
        backoff *= 2
        continue
      of Fail: return false
      of Progress:
        totalbackoff = 0
        continue
      of Complete: return true