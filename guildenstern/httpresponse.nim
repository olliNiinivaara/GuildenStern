{.push hint[DuplicateModuleImport]: off.}
from std/strutils import join
from std/strformat import fmt
{.pop.}

template intermediateflags(): cint = MSG_NOSIGNAL + MSG_DONTWAIT + MSG_MORE
template lastflags(): cint = MSG_NOSIGNAL + MSG_DONTWAIT

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
  # Informs that everything is replied.
  let ret =
    try: send(thesocket, nil, 0, lastflags)
    except CatchableError: -1
  if likely(ret != -1): return Complete
  discard checkSocketState(-1)
  return Fail


proc writeToSocket(text: ptr string, length: int, flags = intermediateflags): SocketState {.inline, gcsafe, raises: [].} =
  if length == 0: return Complete
  var bytessent = 0
  var backoff = initialbackoff
  var totalbackoff = 0
  while true:
    let ret = send(thesocket, unsafeAddr text[bytessent], (length - bytessent).cint, flags)
    if likely(ret > 0):
      bytessent.inc(ret)
      if bytessent == length:
        server.log(DEBUG, "writeToSocket " & $thesocket & ": " & text[0 ..< length])
        return Complete
      continue
    result = checkSocketState(ret)
    if result == TryAgain:
      server.suspend(backoff)
      totalbackoff += backoff
      backoff *= 2
      if totalbackoff > server.sockettimeoutms:
        closeSocket(TimedOut, "didn't write to thesocket")
        return Fail
      continue
    else: return result


proc writeVersion(): SocketState {.inline, gcsafe, raises: [].} =
  {.gcsafe.}: return writeToSocket(unsafeAddr version, 9, intermediateflags)


proc writeCode(code: HttpCode): SocketState {.inline, gcsafe, raises: [].} =
  if code == Http200:
    {.gcsafe.}: return writeToSocket(unsafeAddr http200string, 8, intermediateflags)
  else:
    let codestring = $code & "\c\L" # slow...
    return writeToSocket(unsafeAddr codestring, codestring.len.cint, intermediateflags)


proc tryWriteToSocket(text: ptr string, start: int, length: int, flags = intermediateflags): (SocketState , int) {.inline, gcsafe, raises: [].} =
  assert(not isNil(text) and length > 0)
  result[1] =
    try: send(thesocket, unsafeAddr text[start], length.cint, flags)
    except CatchableError: Excepted.int
  if likely(result[1] > 0):
    if result[1] == length: result[0] = Complete
    else: result[0] = Progress
  else: result[0] =  checkSocketState(result[1])


proc reply*(code: HttpCode): SocketState {.discardable, inline, gcsafe, raises: [].} =
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


proc reply*(code: HttpCode, body: ptr string, lengthstring: string, length: int, headers: ptr string, moretocome: bool): SocketState {.gcsafe, raises: [].} =
  ## One-shot reply to a request
  let finalflag = if moretocome: intermediateflags else: lastflags
  {.gcsafe.}: 
    if unlikely(writeVersion() != Complete): return Fail 
    if unlikely(writeCode(code) != Complete): return Fail

    if (not isNil(headers)) and headers[].len > 0:
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
  ## Start replying to a request (continue with [replyMore] and [replyFinish]).
  ## If you do not know the content-length yet, use [replyStartChunked] instead.
  {.gcsafe.}: 
    if unlikely(writeVersion() != Complete): return Fail 
    if unlikely(writeCode(code) != Complete): return Fail

    if (not isNil(headers)) and headers[].len > 0:
      if writeToSocket(headers, headers[].len) != Complete: return Fail

    if contentlength < 1: return writeToSocket(unsafeAddr longdivider, longdivider.len)

    if (not isNil(headers)) and headers[].len > 0:
      if unlikely(writeToSocket(unsafeAddr shortdivider, shortdivider.len) != Complete): return Fail
    
    if unlikely(writeToSocket(unsafeAddr contentlen, contentlen.len) != Complete): return Fail
    let lengthstring = $contentlength
    if unlikely(writeToSocket(unsafeAddr lengthstring, lengthstring.len) != Complete): return Fail
    return writeToSocket(unsafeAddr longdivider, longdivider.len)


proc reply*(code: HttpCode, body: ptr string, headers: ptr string) {.inline, gcsafe, raises: [].} =
  let length = if isNil(body): 0 else: body[].len
  if likely(reply(code, body, $length, length, headers, false) == Complete): server.log(TRACE, "reply ok")
  else: server.log(INFO, $thesocket & ": reply failed")

proc reply*(code: HttpCode, body: ptr string, headers: openArray[string]) {.inline, gcsafe, raises: [].} =
  let joinedheaders = headers.join("\c\L")
  reply(code, body, unsafeAddr joinedheaders)

proc replyStart*(code: HttpCode, contentlength: int, headers: openArray[string]): SocketState {.inline, gcsafe, raises: [].} =
  let joinedheaders = headers.join("\c\L")
  replyStart(code, contentlength, unsafeAddr joinedheaders)

proc replyMore*(bodypart: ptr string, start: int, partlength: int = -1): (SocketState , int) {.inline, gcsafe, raises: [].} =
  ## Continuation to [replyStart].
  let length = if partlength != -1: partlength else: bodypart[].len
  return tryWriteToSocket(bodypart, start, length)

template reply*(code: HttpCode, headers: openArray[string]) =
  reply(code, nil, headers)

template reply*(body: string) =
  when compiles(unsafeAddr body):
    reply(Http200, unsafeAddr body, nil)
  else:
    let copy = body
    reply(Http200, unsafeAddr copy, nil)

template reply*(code: HttpCode, body: string) =
  when compiles(unsafeAddr body):
    reply(code, unsafeAddr body, nil)
  else:
    let copy = body
    reply(code, unsafeAddr copy, nil)

template reply*(code: HttpCode, body: string, headers: openArray[string]) =
  when compiles(unsafeAddr body):
    reply(code, unsafeAddr body, headers)
  else:
    let copy = body
    reply(code, unsafeAddr copy, headers)

template reply*(body: string, headers: openArray[string]) =
  when compiles(unsafeAddr body):
    reply(Http200, unsafeAddr body, headers)
  else:
    let copy = body
    reply(Http200, unsafeAddr copy, headers)

template replyMore*(bodypart: string): bool =
  when compiles(unsafeAddr bodypart):
    replyMore(unsafeAddr bodypart, 0)
  else:
    let copy = bodypart
    replyMore(unsafeAddr copy, 0)


proc replyStartChunked*(code: HttpCode = Http200, headers: openArray[string] = []): bool {.gcsafe.} =
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


proc replyContinueChunked*(chunk: string): bool {.gcsafe.} =
  var backoff = initialbackoff
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
      server.suspend(backoff)
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


proc replyFinishChunked*(): bool {.gcsafe, discardable.} =
  {.gcsafe.}:
    let delimiter = "0" & longdivider
  if writeToSocket(addr delimiter, delimiter.len) == Fail: return false
  return replyFinish() != Fail

