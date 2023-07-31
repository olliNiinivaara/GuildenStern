const
  DONTWAIT = 0x40.cint
  intermediateflags = MSG_NOSIGNAL + 0x8000 # MSG_MORE
  lastflag = MSG_NOSIGNAL

let
  version = "HTTP/1.1 "
  http200string = "200 OK"
  http204string = "204 No Content"


proc writeVersion*(): SocketState {.inline, gcsafe, raises: [].} =
  {.gcsafe.}:
    let ret = send(http.socketdata.socket, unsafeAddr version[0], 9, intermediateflags)
  if unlikely(ret != 9):
    result = checkSocketState(ret)
    if result == Progress:
      result = Fail
      server.closeSocket(http.socketdata, ProtocolViolated, "")
  else: return Complete


proc writeCode*(code: HttpCode): SocketState {.inline, gcsafe, raises: [].} =
  var ret: int
  try:
    if likely(code == Http200):
      {.gcsafe.}: ret = send(http.socketdata.socket, unsafeAddr http200string[0], 6, intermediateflags)
    elif code == Http204:
      {.gcsafe.}: ret = send(http.socketdata.socket, unsafeAddr http204string[0], 14, 0)
    else:
      let codestring = $code # slow...
      ret = send(http.socketdata.socket, unsafeAddr codestring[0], codestring.len.cint, 0)
  except CatchableError:
    ret = Excepted.int
  result = checkSocketState(ret)
  if likely(result == Progress): result = Complete
  server.log(DEBUG, "writeCode " & $http.socketdata.socket & " " & $code & ": " & $result)


proc writeToSocketNonblocking*(text: ptr string, start: int, length: int, flags = intermediateflags): (SocketState , int) {.inline, gcsafe, raises: [].} =
  assert(text != nil and length > 0)
  result[1] =
    try: send(http.socketdata.socket, unsafeAddr text[start], length.cint, flags + DONTWAIT)
    except CatchableError: Excepted.int
  if likely(result[1] > 0):
    if result[1] == length: result[0] = Complete
    else: result[0] = Progress
  else: result[0] =  checkSocketState(result[1])


proc replyFinish*(): SocketState {.discardable, inline, gcsafe, raises: [].} =
  let ret =
    try: send(http.socketdata.socket, nil, 0, lastflag)
    except CatchableError: Excepted.int
  if likely(ret != -1): return Complete
  discard checkSocketState(-1)
  return Fail


proc writeToSocketBlocking*(text: ptr string, length: int, flags = intermediateflags): SocketState {.inline, gcsafe, raises: [].} =
  if length == 0: return Complete
  var bytessent = 0
  var backoff = 1
  var totalbackoff = 0
  while bytessent < length:
    let ret =
      try: send(http.socketdata.socket, unsafeAddr text[bytessent], (length - bytessent).cint, flags + DONTWAIT)
      except CatchableError: Excepted.int
    if likely(ret > 0):
      bytessent.inc(ret)
      continue
    result = checkSocketState(ret)
    if result == TryAgain:
      sleep(backoff)
      totalbackoff += backoff
      backoff *= 2
      if totalbackoff > server.blockingsendtimeoutms:
        server.closeSocket(http.socketdata, TimedOut, "didn't write to socket")
        return Fail
      continue
    else: return result    
  if text[0] != '\c':
    server.log(DEBUG, "writeToSocket " & $http.socketdata.socket & ": " & text[0 ..< length])
  return Complete


let
  shortdivider = "\c\L"
  longdivider = "\c\L\c\L"
  contentlen = "Content-Length: "
  zerocontent = "Content-Length: 0\c\L"


proc replyCode*(code: HttpCode = Http200): SocketState {.discardable, inline, gcsafe, raises: [].} =
  if unlikely(writeVersion() != Complete) : return
  if unlikely(writeCode(code) != Complete): return
  {.gcsafe.}:
      if code == Http204: return writeToSocketBlocking(unsafeAddr longdivider, longdivider.len, lastflag)
      else:
        if unlikely(writeToSocketBlocking(unsafeAddr shortdivider, shortdivider.len) != Complete): return Fail
        if unlikely(writeToSocketBlocking(unsafeAddr zerocontent, zerocontent.len) != Complete): return Fail
        return writeToSocketBlocking(unsafeAddr shortdivider, shortdivider.len, lastflag)
      

proc reply*(code: HttpCode, body: ptr string, lengthstring: string, length: int, headers: ptr string, moretocome: bool): SocketState {.gcsafe, raises: [].} =
  if body == nil and headers == nil: return replyCode(code)
  let finalflag = if moretocome: intermediateflags else: lastflag
  {.gcsafe.}: 
    if unlikely(writeVersion() != Complete): return Fail 
    if unlikely(writeCode(code) != Complete): return Fail
    if unlikely(writeToSocketBlocking(unsafeAddr shortdivider, shortdivider.len) != Complete): return Fail

    if headers != nil and headers[].len > 0:
      if writeToSocketBlocking(headers, headers[].len) != Complete: return Fail
      if writeToSocketBlocking(unsafeAddr shortdivider, shortdivider.len) != Complete: return Fail

    if code == Http101 or code == Http304:      
      return writeToSocketBlocking(unsafeAddr shortdivider, shortdivider.len, finalflag)
      
    if length < 1:      
      if writeToSocketBlocking(unsafeAddr zerocontent, zerocontent.len) != Complete: return Fail
      return writeToSocketBlocking(unsafeAddr longdivider, longdivider.len, finalflag)
      
    if writeToSocketBlocking(unsafeAddr contentlen, contentlen.len) != Complete: return Fail
    if writeToSocketBlocking(unsafeAddr lengthstring, lengthstring.len) != Complete: return Fail
    if writeToSocketBlocking(unsafeAddr longdivider, longdivider.len) != Complete: return Fail
    return writeToSocketBlocking(body, length, finalflag)


proc replyStart*(code: HttpCode, contentlength: int, headers: ptr string = nil): SocketState {.inline, gcsafe, raises: [].} =
  {.gcsafe.}: 
    if unlikely(writeVersion() != Complete): return Fail 
    if unlikely(writeCode(code) != Complete): return Fail
    if unlikely(writeToSocketBlocking(unsafeAddr shortdivider, shortdivider.len) != Complete): return Fail

    if headers != nil and headers[].len > 0:
      if writeToSocketBlocking(headers, headers[].len) != Complete: return Fail
      if writeToSocketBlocking(unsafeAddr shortdivider, shortdivider.len) != Complete: return Fail
    
    if unlikely(writeToSocketBlocking(unsafeAddr contentlen, contentlen.len) != Complete): return Fail
    let lengthstring = $contentlength
    if unlikely(writeToSocketBlocking(unsafeAddr lengthstring, lengthstring.len) != Complete): return Fail
    return writeToSocketBlocking(unsafeAddr longdivider, longdivider.len)


proc reply*(code: HttpCode, body: ptr string = nil, headers: ptr string = nil) {.inline, gcsafe, raises: [].} =
  let length = if body == nil: 0 else: body[].len
  if likely(reply(code, body, $length, length, headers, false) == Complete): server.log(TRACE, "reply ok")
  else: server.log(INFO, $http.socketdata.socket & ": reply failed")

proc reply*(code: HttpCode, body: ptr string, headers: openArray[string]) {.inline, gcsafe, raises: [].} =
  let joinedheaders = headers.join("\c\L")
  reply(code, body, unsafeAddr joinedheaders)

proc replyStart*(code: HttpCode, contentlength: int, headers: openArray[string]): SocketState {.inline, gcsafe, raises: [].} =
  let joinedheaders = headers.join("\c\L")
  replyStart(code, contentlength, unsafeAddr joinedheaders)

proc replyMore*(bodypart: ptr string, start: int, partlength: int = -1): (SocketState , int) {.inline, gcsafe, raises: [].} =
  let length = if partlength != -1: partlength else: bodypart[].len
  return writeToSocketNonBlocking(bodypart, start, length)

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

template reply*(code: HttpCode, body: string) =
  when compiles(unsafeAddr body):
    reply(code, unsafeAddr body, nil)
  else: {.fatal: "posix.send requires taking pointer to body, but body has no address".}

template replyMore*(bodypart: string): bool =
  when compiles(unsafeAddr bodypart):
    replyMore(unsafeAddr bodypart, 0)
  else: {.fatal: "posix.send requires taking pointer to bodypart, but bodypart has no address".}
