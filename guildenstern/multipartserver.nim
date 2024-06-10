## Multipart/formdata server
## 
## see examples/multiparttest.nim for a concrete example.
##

import std/strtabs
from strutils import find, parseInt, startsWith, toLowerAscii
export strtabs
import httpserver
export httpserver

type
  PartState* = enum
    ## State of the request delivery inside the [receiveParts] iterator
    HeaderReady ## New part starts. [parseContentDisposition] is your friend here. For accessing other fields, use http.headers
    BodyChunk ## More data for the current part body has arrived
    BodyReady ## Current part is received
    Failed ## See contents of chunk for potential additional info
    Completed ## That's it

  MultipartContext = ref object of HttpContext
    boundary: string
    partcache: string
    partlen: int
    headercache: string
    headerlen: int
    inheader: bool
    parsepartheaders: bool
    
  
template multipart(): untyped = MultipartContext(socketcontext)

proc processHeader(c: char): PartState =
  if multipart.headerlen + 1 > server.maxheaderlength:
    closeSocket(ProtocolViolated, "header of part exceeded maximum allowed size")
    return Failed
  multipart.headercache[multipart.headerlen] = c
  multipart.headerlen += 1
  if multipart.headerlen > 20 and c == '\L':
    if multipart.headercache[multipart.headerlen - 2] == '\c' and
    multipart.headercache[multipart.headerlen - 3] == '\L' and
    multipart.headercache[multipart.headerlen - 4] == '\c':
      multipart.inheader = false
      return HeaderReady
    else: return Completed
  else: return Completed


proc atBoundary(): bool =
  for i in countdown(multipart.boundary.len - 1, 0):
    if multipart.boundary[i] != multipart.partcache[multipart.partlen - 1 - multipart.boundary.len - 1 + i]: return false
  true


proc processPart(c: char): PartState =
  multipart.partcache[multipart.partlen] = c
  multipart.partlen += 1
  if multipart.partlen < multipart.boundary.len: return Completed
  if atBoundary():
    multipart.inheader = true
    return BodyReady
  else:
    if multipart.partlen < multipart.partcache.len: return Completed
    return BodyChunk


iterator processChunk(chunk: string): (PartState , string) =
  for c in chunk:
    if multipart.inheader:
      let headerstate = processHeader(c)
      if headerstate == Failed:
        yield(Failed, "header of part exceeded maximum allowed size")
        break
      elif headerstate == HeaderReady:
        yield(HeaderReady, multipart.headercache[0 .. multipart.headerlen - 5])
        multipart.headerlen = 0
      else: yield(Completed , "")
    else:
      let partstate = processPart(c)
      if partstate == Failed:
        yield(Failed, "part processing failed")
        break
      elif partstate == BodyChunk:
        yield(BodyChunk , multipart.partcache[0 .. multipart.partlen - multipart.boundary.len - 1])
        for i in 0 .. multipart.boundary.len - 1: multipart.partcache[i] = multipart.partcache[multipart.partlen - multipart.boundary.len + i]
        multipart.partlen = multipart.boundary.len
      elif partstate == BodyReady:
        let last = multipart.partlen - multipart.boundary.len - 5
        yield(BodyChunk , multipart.partcache[0 .. last])
        multipart.partlen = 0
        yield(BodyReady , "")
      else: yield(Completed , "")


proc parseHeader(header: string) =
  var value = false
  var current: (string, string) = ("", "")
  var i = 0
  try:
    for key in http.headers.keys: http.headers[key].setLen(0)
  except:
    echo "header key error, should never happen"
  while i < header.len:
    case header[i]
    of '\c': discard
    of ':':
      if value: current[1].add(':')
      value = true
    of ' ':
      if value:
        if current[1].len != 0: current[1].add(header[i])
      else: current[0].add(header[i])
    of '\l':
      if not current[0].startsWith("--"): http.headers[current[0]] = current[1]
      value = false
      current = ("", "")
    else:
      if value: current[1].add(header[i])
      else: current[0].add((header[i]).toLowerAscii())
    i.inc
  if current[0].len > 1 and not current[0].startsWith("--"): http.headers[current[0]] = current[1]

    
iterator receiveParts*(parsepartheaders: bool = true): (PartState , string) =
  ## Iterator for streaming in multipart/formdata
  multipart.headerlen = 0
  multipart.partlen = 0
  multipart.inheader = true
  var failed = false
  var backoff = 4
  var totalbackoff = 0

  var originalheaderfields = newSeq[string]() 
  if parsepartheaders:
    for key in http.headers.keys(): originalheaderfields.add(key)
    http.headers.clear()

  for (state , chunk) in receiveStream():
    case state:
      of TryAgain:
        suspend(backoff)
        totalbackoff += backoff
        if totalbackoff > server.sockettimeoutms:
          closeSocket(TimedOut, "didn't stream all contents from socket")
          yield (Failed , "TimedOut")
          break
        backoff *= 2
        continue
      of Fail:
        yield (Failed , "socket failure")
        break
      of Progress:
        totalbackoff = 0
        for (partstate , part) in processChunk(chunk):
          case partstate:
            of Failed:
              yield(Failed , part)
              failed = true
            of HeaderReady:
              if parsepartheaders: parseHeader(part)
              yield(HeaderReady, part)
            of BodyChunk:
              if part.len != 0: yield(BodyChunk, part)
            of BodyReady:
              yield(BodyReady, part)
            of Completed: # nothing to deliver yet
              discard
      of Complete:
        yield(Completed , "")
    if failed: break

  if parsepartheaders:
    for key in originalheaderfields: http.headers[key] = ""


proc parseContentDisposition*(): (string , string) {.raises:[].} =
  ## Returns values of name and filename properties of the content-disposition header field 
  let value = http.headers.getOrDefault("content-disposition")
  if value.len < 18: return
  let nameend = value.find('"', 18)
  if nameend == -1: return
  result[0] = value[17 ..< nameend]
  if result[0] == "": return
  let filenamestart = value.find('"', nameend + 1) + 1
  if filenamestart == 0: return
  result[1] = value[filenamestart .. value.len - 2]


proc handleMultipartInitialization(gserver: GuildenServer) =
  socketcontext = new MultipartContext
  handleHttpThreadInitialization(gserver)
  multipart.headercache = newString(HttpServer(gserver).maxheaderlength + 1)
  multipart.partcache = newString(HttpServer(gserver).bufferlength + 1)


proc handleMultipartRequest(data: ptr SocketData) {.gcsafe, nimcall, raises: [].} =
  let socketdata = data[]
  let socketint = socketdata.socket.int
  if unlikely(socketint == -1): return
  prepareHttpContext(addr socketdata)
  if not readHeader(): return        
  if not parseRequestLine(): return
  let contenttype = http.headers.getOrDefault("content-type")
  if not contenttype.startsWith("multipart/form-data; boundary="):
    closeSocket(ProtocolViolated, "Multipart request with wrong content-type (" & contenttype & ") received from socket " & $socketint)
    return
  multipart.boundary = "--" & contenttype[30 .. ^1] # last boundary's extra -- is just ignored
  if unlikely(multipart.boundary.len > server.bufferlength - 1): server.log(ERROR, "bufferlength too small, even part boundary does not fit")
  server.log(DEBUG, "Started multipart streaming with chunk of length " & $http.requestlen & " from socket " & $socketint)
  {.gcsafe.}: server.requestCallback()


proc newMultipartServer*(onrequestcallback: proc(){.gcsafe, nimcall, raises: [].}, loglevel = LogLevel.WARN, headerfields: openArray[string] = []): HttpServer =
  ## Note: headerfields concern only the whole request, not part headers
  result = new HttpServer
  result.internalThreadInitializationCallback = handleMultiPartInitialization
  var fields = newSeq[string](headerfields.len + 1)
  fields.add(headerfields)
  if not fields.contains("content-type"): fields.add("content-type")
  result.initHttpServer(loglevel, true, Streaming, fields)
  result.handlerCallback = handleMultipartRequest
  result.requestCallback = onrequestcallback
