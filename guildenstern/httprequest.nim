proc parseMethod*(): bool =
  if unlikely(http.requestlen < 13):
    server.log(WARN, "too short request: " & http.request)
    closeSocket(ProtocolViolated, "")
    return false
  while http.methlen < http.requestlen and http.request[http.methlen] != ' ': http.methlen.inc
  if unlikely(http.methlen == http.requestlen):
    server.log(WARN, "http method missing")
    closeSocket(ProtocolViolated, "")
    return false
  if unlikely(http.request[0 .. 1] notin ["GE", "PO", "HE", "PU", "DE", "CO", "OP", "TR", "PA"]):
    server.log(WARN, "invalid http method: " & http.request[0 .. 12])
    closeSocket(ProtocolViolated, "")
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
    (closeSocket(ProtocolViolated, ""); return false)

  if unlikely(http.request[http.uristart + http.urilen + 1] != 'H' or http.request[http.uristart + http.urilen + 8] != '1'):
    server.log(WARN, "request not HTTP/1.1: " & http.request[http.uristart + http.urilen + 1 .. http.uristart + http.urilen + 8])
    (closeSocket(ProtocolViolated, ""); return false)
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
  proc getBodyview*(http: HttpHandler): openArray[char] =
    if http.bodystart < 1: return http.request.toOpenArray(0, -1)
    else: return http.request.toOpenArray(http.bodystart, http.requestlen - 1)


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
  var expectedlength = server.maxrequestlength + 1
  var backoff = 1
  var totalbackoff = 0
  while true:
    if unlikely(shuttingdown): return false
    let ret = recv(http.socketdata.socket, addr http.request[http.requestlen], expectedlength - http.requestlen, MSG_DONTWAIT)
    if unlikely(ret < 1):
      let state = checkSocketState(ret)
      if likely(state == TryAgain):
        suspend(backoff)
        totalbackoff += backoff
        backoff *= 2
        if totalbackoff > server.sockettimeoutms:
          closeSocket(TimedOut, "didn't read from socket")
          return false
        continue
      if state == Fail: return false

    let previouslen = http.requestlen
    http.requestlen += ret

    if unlikely(http.requestlen >= server.maxrequestlength):
      closeSocket(ProtocolViolated, "recvHttp: Max request size exceeded")
      return false

    if http.requestlen == expectedlength: break

    if not isHeaderreceived(previouslen, http.requestlen):
      if http.requestlen >= server.maxheaderlength:
        closeSocket(ProtocolViolated, "recvHttp: Max header size exceeded" )
        return false
      continue

    let contentlength = getContentLength()
    if contentlength == 0: return true
    expectedlength = http.bodystart + contentlength
    if http.requestlen == expectedlength: break
  server.log(DEBUG, $server.port & "/" & $http.socketdata.socket & ": " & http.request[http.bodystart .. http.bodystart + http.requestlen - 1])
  true


proc receiveHeader*(): bool {.gcsafe, raises:[].} =
  var backoff = 1
  var totalbackoff = 0
  while true:
    if unlikely(shuttingdown): return false
    let ret = recv(http.socketdata.socket, addr http.request[http.requestlen], 1 + server.maxheaderlength - http.requestlen, MSG_DONTWAIT)
    if unlikely(ret < 1):
      let state = checkSocketState(ret)
      if (likely)state == TryAgain:
        suspend(backoff - 1)
        totalbackoff += backoff
        backoff = backoff shl 2
        if unlikely(totalbackoff > server.sockettimeoutms):
          closeSocket(TimedOut, "didn't read from socket in " & $server.sockettimeoutms & " ms")
          return false
        continue
      if state == Fail: return false
    http.requestlen += ret
    if http.requestlen > server.maxheaderlength:
      closeSocket(ProtocolViolated, "receiveHeader: Max header size exceeded")
      return false
    if http.request[http.requestlen-4] == '\c' and http.request[http.requestlen-3] == '\l' and
     http.request[http.requestlen-2] == '\c' and http.request[http.requestlen-1] == '\l': break
  return http.requestlen > 0



