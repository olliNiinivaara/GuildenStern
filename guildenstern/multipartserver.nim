import std/[times, monotimes, strtabs]
from posix import recv, SocketHandle
from strutils import find, parseInt, startsWith
from strformat import fmt
export strtabs
import httpserver
export httpserver

type
  FileState = enum Before, In, After

  StreamState = enum
    StreamFail, StreamTryAgain, StreamProgress, StreamComplete, StreamPartStart

  PartStreamState* = enum HeaderReady, PartChunk,  PartReady, Failed, Completed


when defined(nimdoc):
  type StreamingContext* = ref object of HttpContext
else:
  type 
    MultipartContext* = ref object of HttpContext
      boundary: string
      filestate: FileState
      partcache: string
      headercache: string
      headerlen: int
    
  
proc isMultipartContext*(): bool = return socketcontext is MultipartContext

template multipart*(): untyped =
  ## Casts the socketcontext thread local variable into a MultipartContext
  MultipartContext(socketcontext)


proc processPart(chunk: string): (StreamState , string) =
  assert multipart.filestate != After
  let endboundarylength = multipart.boundary.len + 2
  multipart.partcache.add(chunk)
  if multipart.partcache.len < endboundarylength:
    # echo "cache might contain part of boundary, more data needed"
    return (StreamTryAgain , "")
  let boundarystartsat = multipart.partcache.find(multipart.boundary)
  let boundaryfreeend = multipart.partcache.len - multipart.boundary.len - 1
  if boundarystartsat == -1:
    # echo "no boundaries found, end of cache might contain part of forthcoming boundary"
    result[1] = multipart.partcache[0 .. boundaryfreeend]
    multipart.partcache =  multipart.partcache[boundaryfreeend .. ^1]
    if multipart.filestate == Before:
      result[0] = StreamPartStart
      multipart.filestate = In
    else: result[0] = StreamProgress
  else:
    # boundary found in cache
    let boundaryendedbefore = boundarystartsat + multipart.boundary.len
    if multipart.partcache.len < endboundarylength or multipart.partcache[boundaryendedbefore .. boundaryendedbefore + 1] != "--":
      # echo "startboundary found"
      result[1] = multipart.partcache[0 ..< boundarystartsat]
      multipart.partcache = multipart.partcache[boundaryendedbefore .. ^1]
      if multipart.filestate == Before:
        result[0] = StreamPartStart
      else:
        result[0] = StreamProgress
      multipart.filestate = Before
    else:
      # echo "end boundary found"
      result[1] = multipart.partcache[0 ..< boundarystartsat]
      multipart.partcache = ""
      if multipart.filestate == Before:
        result[0] = StreamPartStart
      else: result[0] = StreamComplete


iterator coarseReceiveParts(): (StreamState , string) {.gcsafe, raises: [].} =
  # assert(multipart.partcache == "readytostart", "forgot to call startReceiveMultipart?")
  multipart.partcache = ""
  for (state , chunk) in receiveStream():
    #[if getMonoTime() - multipart.start > multipart.timeout:
      closeSocket(TimedOut, "reached giveupSecs = " & $multipart.timeout)
      yield(StreamState.Fail, "")
      break]#
    case state:
      of TryAgain:
        suspend(10) #multipart.waitms)
        yield(StreamTryAgain, "")
        # backoff timeout
      of Fail: yield(StreamFail, "")
      of Progress:
        if multipart.filestate == After: continue
        let (s , r) = processPart(chunk)
        if s == StreamProgress: yield(StreamProgress, r)
        elif s == StreamComplete:
          if r.len > 0: yield(StreamProgress, r)
          multipart.filestate = After
        elif s == StreamPartStart:
          yield(StreamPartStart, "")
          if r.len > 0: yield(StreamProgress, r)
      of Complete:
        if multipart.filestate == After: yield(StreamComplete, "")
        else:
          while true:
            let (s , r) = processPart("")
            if s == StreamComplete:
              if r.len > 0: yield(StreamProgress, r)
              yield(StreamComplete, "")
              break
            elif s == StreamPartStart:
              if r.len == 0:
                yield(StreamComplete, "")
                break
              yield(StreamPartStart, "")
              yield(StreamProgress, r)
            else:
              yield(StreamComplete, "")
              break


iterator receiveParts*(): (PartStreamState , string) =
  var
    trials = 0
    inheader = true
  multipart.headerlen = 0
  for (recvstate , recvchunk) in coarseReceiveParts():
    case recvstate:
      of StreamTryAgain: # give up after x seconds of inactivity
        trials += 1
        if trials > 100:
          closeSocket(TimedOut, "syy")
          yield (Failed , "syy")
          break
      of StreamFail:
        yield (Failed , "")
        break
      #[of HeaderStart:
        if not inheader: yield(PartReady , "")
        headerlen = 0
        inheader = true]#
      of StreamPartStart:
        inheader = true
        yield(HeaderReady , multipart.headercache[0 ..< multipart.headerlen])
        multipart.headerlen = 0
      of StreamProgress:
        if inheader:
          if multipart.headerlen + recvchunk.len > server.maxheaderlength:
            closeSocket(ProtocolViolated, "syy")
            yield(Failed , "syy")
            break
          for i in 0 ..< recvchunk.len: multipart.headercache[multipart.headerlen + i] = recvchunk[i]
          multipart.headerlen += recvchunk.len
        else:
          yield(PartChunk , recvchunk)
      of StreamComplete:
        yield(PartReady , "") 
        yield(Completed , "")       


when not defined(nimdoc):
  proc handleMultipartRequest(data: ptr SocketData) {.gcsafe, nimcall, raises: [].} =
    let socketdata = data[]
    let socketint = socketdata.socket.int
    if unlikely(socketint == -1): return
    if (unlikely)socketcontext == nil:
      socketcontext = new MultipartContext 
    prepareHttpContext(addr socketdata)
    if unlikely(multipart.headercache.len != server.maxheaderlength + 1):
      multipart.headercache = newString(server.maxheaderlength + 1)
    if not readHeader(): return        
    if not parseRequestLine(): return
    let contenttype = http.headers.getOrDefault("content-type")
    if not contenttype.startsWith("multipart/form-data; boundary="):
      # remember logging...
      return
    multipart.filestate = Before
    multipart.boundary = "--" & contenttype[30 .. ^1] # START BOUNDARY
    server.log(DEBUG, "Started multipart streaming with chunk of length " & $http.requestlen & " from socket " & $socketint)
    {.gcsafe.}: server.requestCallback()


proc newMultipartServer*(onrequestcallback: proc(){.gcsafe, nimcall, raises: [].}, loglevel = LogLevel.WARN, headerfields: openArray[string] = ["content-type"]): HttpServer =
  result = new HttpServer
  when not defined(nimdoc):
    var fields = newSeq[string](headerfields.len + 1)
    fields.add(headerfields)
    if not fields.contains("content-type"): fields.add(headerfields)
    result.initHttpServer(loglevel, true, Streaming, fields)
    result.handlerCallback = handleMultipartRequest
    result.requestCallback = onrequestcallback
