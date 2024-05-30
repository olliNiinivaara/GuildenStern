{.push hint[DuplicateModuleImport]: off.}
from std/strutils import join
from std/strformat import fmt
{.pop.}

when not defined(nimdoc):
  const
    intermediateflags = MSG_NOSIGNAL + MSG_DONTWAIT + MSG_MORE
    lastflags = MSG_NOSIGNAL + MSG_DONTWAIT
  

let
  version = "HTTP/1.1 "
  http200string = "200 OK\c\L"
  http200nocontent = "HTTP/1.1 200 OK\c\LContent-Length: 0\c\L\c\L"
  http200nocontentlen = 38
  http204string = "HTTP/1.1 204 No Content\c\L\c\L"
  http204stringlen = 27
  shortdivider* = "\c\L"
  longdivider* = "\c\L\c\L"
  contentlen = "Content-Length: "
  zerocontent = "Content-Length: 0\c\L"


proc replyFinish*(): SocketState {.discardable, inline, gcsafe, raises: [].} =
  when defined(nimdoc): discard
  else:
    let ret =
      try: send(http.socketdata.socket, nil, 0, lastflags)
      except CatchableError: Excepted.int
    if likely(ret != -1): return Complete
    discard checkSocketState(-1)
    return Fail


when not defined(nimdoc):
  proc writeToSocket*(text: ptr string, length: int, flags = intermediateflags): SocketState {.inline, gcsafe, raises: [].} =
    if length == 0: return Complete
    var bytessent = 0
    var backoff = 1
    var totalbackoff = 0
    while true:
      let ret = send(http.socketdata.socket, unsafeAddr text[bytessent], (length - bytessent).cint, flags)
      if likely(ret > 0):
        bytessent.inc(ret)
        if bytessent == length:
          server.log(DEBUG, "writeToSocket " & $http.socketdata.socket & ": " & text[0 ..< length])
          return Complete
        continue
      result = checkSocketState(ret)
      if result == TryAgain:
        suspend(backoff)
        totalbackoff += backoff
        backoff *= 2
        if totalbackoff > server.sockettimeoutms:
          closeSocket(TimedOut, "didn't write to socket")
          return Fail
        continue
      else: return result


  proc writeVersion*(): SocketState {.inline, gcsafe, raises: [].} =
    {.gcsafe.}: return writeToSocket(unsafeAddr version, 9, intermediateflags)


  proc writeCode*(code: HttpCode): SocketState {.inline, gcsafe, raises: [].} =
    if code == Http200:
      {.gcsafe.}: return writeToSocket(unsafeAddr http200string, 8, intermediateflags)
    else:
      let codestring = $code & "\c\L" # slow...
      return writeToSocket(unsafeAddr codestring, codestring.len.cint, intermediateflags)


  proc tryWriteToSocket*(text: ptr string, start: int, length: int, flags = intermediateflags): (SocketState , int) {.inline, gcsafe, raises: [].} =
    assert(text != nil and length > 0)
    result[1] =
      try: send(http.socketdata.socket, unsafeAddr text[start], length.cint, flags)
      except CatchableError: Excepted.int
    if likely(result[1] > 0):
      if result[1] == length: result[0] = Complete
      else: result[0] = Progress
    else: result[0] =  checkSocketState(result[1])


proc reply*(code: HttpCode): SocketState {.discardable, inline, gcsafe, raises: [].} =
  if unlikely(http.socketdata.socket.int in [0, INVALID_SOCKET.int]):
    # TODO: why does log template not work here?
    echo "cannot reply to closed socket"
    return
  when defined(nimdoc): discard
  else:
    {.gcsafe.}:
      if code == Http200:
        return writeToSocket(unsafeAddr http200nocontent, http200nocontentlen, lastflags)
      elif code == Http204:
        return writeToSocket(unsafeAddr http204string, http204stringlen, lastflags)
      else:
        if unlikely(writeVersion() != Complete) : return Fail
        if unlikely(writeCode(code) != Complete): return Fail
        if unlikely(writeToSocket(unsafeAddr zerocontent, zerocontent.len) != Complete): return Fail
        return writeToSocket(unsafeAddr shortdivider, shortdivider.len, lastflags)


when not defined(nimdoc):
  proc reply*(code: HttpCode, body: ptr string, lengthstring: string, length: int, headers: ptr string, moretocome: bool): SocketState {.gcsafe, raises: [].} =
    if unlikely(http.socketdata.socket.int in [0, INVALID_SOCKET.int]):
      echo "cannot reply to closed socket"
      return
    let finalflag = if moretocome: intermediateflags else: lastflags
    {.gcsafe.}: 
      if unlikely(writeVersion() != Complete): return Fail 
      if unlikely(writeCode(code) != Complete): return Fail

      if headers != nil and headers[].len > 0:
        if writeToSocket(headers, headers[].len) != Complete: return Fail
        if writeToSocket(unsafeAddr shortdivider, shortdivider.len) != Complete: return Fail

      if code == Http101 or code == Http304:      
        return writeToSocket(unsafeAddr shortdivider, shortdivider.len, lastflags)
        
      if length < 1:      
        if writeToSocket(unsafeAddr zerocontent, zerocontent.len) != Complete: return Fail
        return writeToSocket(unsafeAddr longdivider, longdivider.len, lastflags)
        
      if writeToSocket(unsafeAddr contentlen, contentlen.len) != Complete: return Fail
      if writeToSocket(unsafeAddr lengthstring, lengthstring.len) != Complete: return Fail
      if writeToSocket(unsafeAddr longdivider, longdivider.len) != Complete: return Fail
      return writeToSocket(body, length, finalflag)


proc replyStart*(code: HttpCode, contentlength: int, headers: ptr string = nil): SocketState {.inline, gcsafe, raises: [].} =
  if unlikely(http.socketdata.socket.int in [0, INVALID_SOCKET.int]):
    echo "cannot reply to closed socket"
    return
  when defined(nimdoc): discard
  else:
    {.gcsafe.}: 
      if unlikely(writeVersion() != Complete): return Fail 
      if unlikely(writeCode(code) != Complete): return Fail

      if headers != nil and headers[].len > 0:
        if writeToSocket(headers, headers[].len) != Complete: return Fail

      if contentlength < 1: return writeToSocket(unsafeAddr longdivider, longdivider.len)

      if headers != nil and headers[].len > 0:
        if unlikely(writeToSocket(unsafeAddr shortdivider, shortdivider.len) != Complete): return Fail
      
      if unlikely(writeToSocket(unsafeAddr contentlen, contentlen.len) != Complete): return Fail
      let lengthstring = $contentlength
      if unlikely(writeToSocket(unsafeAddr lengthstring, lengthstring.len) != Complete): return Fail
      return writeToSocket(unsafeAddr longdivider, longdivider.len)


proc reply*(code: HttpCode, body: ptr string, headers: ptr string) {.inline, gcsafe, raises: [].} =
  when defined(nimdoc): discard
  else:
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
  when not defined(nimdoc): return tryWriteToSocket(bodypart, start, length)

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

template reply*(body: string, headers: openArray[string]) =
  when compiles(unsafeAddr body):
    reply(Http200, unsafeAddr body, headers)
  else: {.fatal: "posix.send requires taking pointer to body, but body has no address".}

template replyMore*(bodypart: string): bool =
  when compiles(unsafeAddr bodypart):
    replyMore(unsafeAddr bodypart, 0)
  else: {.fatal: "posix.send requires taking pointer to bodypart, but bodypart has no address".}


proc replyStartChunked*(code: HttpCode = Http200, headers: openArray[string] = []): bool =
  ## Starts replying http response as `Transfer-encoding: chunked`.
  ## Mainly for sending dynamic data, where Content-length header cannot be set.
  ## 
  ## Continue response with calls to `replyContinueChunked`.
  ##
  ## End response with `replyContinueChunked`.
  ## 
  ## See examples/replychunkedtest.nim for a concrete example.
  ## 
  return replyStart(code, -1, ["Transfer-Encoding: chunked"]) != Fail


proc replyContinueChunked*(chunk: string): bool =
  var backoff = 4
  var totalbackoff = 0
  var delivered = 0
  try:
    {.gcsafe.}:
      let delimiter = fmt"{chunk.len:X}" & shortdivider
    if writeToSocket(addr delimiter, delimiter.len) == Fail: return false
  except: return false
  while true:
    if shuttingdown:
      closeSocket()
      return false
    let (state , len) = tryWriteToSocket(addr chunk, delivered, chunk.len - delivered)
    delivered += len
    if state == Fail: return false
    elif state == TryAgain:
      suspend(backoff)
      totalbackoff += backoff
      if totalbackoff > server.sockettimeoutms:
        closeSocket(TimedOut, "didn't write a chunk in time")
        return false
      backoff *= 2
      continue
    elif state == Complete or delivered == chunk.len:
      {.gcsafe.}:
        if writeToSocket(addr shortdivider, shortdivider.len) == Fail: return false
      return true


proc replyFinishChunked*(): bool {.discardable.} =
  when defined(nimdoc): discard
  else:
    {.gcsafe.}:
      let delimiter = "0" & longdivider
    if writeToSocket(addr delimiter, delimiter.len) == Fail: return false
    return replyFinish() != Fail

