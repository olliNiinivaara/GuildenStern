from streams import StringStream, newStringStream, getPosition, setPosition, write
from os import osLastError, osErrorMsg, OSErrorCode, sleep
from posix import recv
import locks
import guildenserver, private/[httpout]
export replyCode, replyEmpty #, replyHeaders


type
  HeaderSlurper* = ref object of Handler
    headerdata*: StringStream
    senddata*: StringStream


const
  MaxSlurperHeaderLength* {.intdefine.} = 1000
  MaxSlurperResponseLength* {.intdefine.} = 100000
 
var
  HeaderSlurping: HandlerType
  lock: Lock
  httpRequestHandlerCallbacks: array[256, proc(handler: HeaderSlurper) {.gcsafe, raises: [].}]
  httpErrorHandlerCallbacks: array[256, proc(handler: Handler) {.gcsafe, raises: [].}]
  headerslurpers: array[MaxHandlerCount, HeaderSlurper]


proc newHeaderSlurper(): HeaderSlurper =
  result = new HeaderSlurper
  result.handlertype = HeaderSlurping
  result.headerdata = newStringStream()
  result.headerdata.data.setLen(MaxSlurperHeaderLength)
  result.senddata = newStringStream()
  

proc getHeaderSlurper(): HeaderSlurper {.inline, raises: [].} =
  {.gcsafe.}:
    withLock(lock):
      var i = 0
      result = headerslurpers[i]
      while result == nil or result.ishandling:
        if result == nil:
          result = newHeaderSlurper()
          headerslurpers[i] = result          
          break
        i += 1
        if i == MaxHandlerCount:
          sleep(20)
          i = 0
        result = headerslurpers[i]
      result.ishandling = true
    result.currentexceptionmsg.setLen(0)
    try:
      result.headerdata.setPosition(0)
      result.senddata.setPosition(0)
    except: (echo "Nim internal error"; return)


{.push checks: off.}

template isFinished: bool =
  h.headerdata.data[recvdatalen-4] == '\c' and h.headerdata.data[recvdatalen-3] == '\l' and h.headerdata.data[recvdatalen-2] == '\c' and h.headerdata.data[recvdatalen-1] == '\l'
  
proc slurpHeader*(h: HeaderSlurper): bool {.gcsafe, raises:[] .} =
  var recvdatalen = 0
  var trials = 0
  while true:
    let ret = recv(h.socket, addr h.headerdata.data[recvdatalen], MaxSlurperHeaderLength + 1, 0.cint)
    if h.gs.serverstate == Shuttingdown: return false
    if ret == 0:
      let lastError = osLastError().int    
      if lastError == 0: return recvdatalen > 0
      if lastError == 2 or lastError == 9 or lastError == 104: return false
      trials.inc
      if trials <= 3: 
        sleep(100 + trials * 100)
        continue
    if ret == -1:
      let lastError = osLastError().int
      h.currentexceptionmsg = "slurp in: " & $lastError & " " & osErrorMsg(OSErrorCode(lastError))
      return false
    if trials >= 4:
      h.currentexceptionmsg = "slurp in: receiving stalled"
      return false
      
    recvdatalen += ret
    if recvdatalen >= MaxSlurperHeaderLength + 1:
      h.currentexceptionmsg = "slurp in: Max request size exceeded"
      return false
    if isFinished: break
  try:
    h.headerdata.setPosition(recvdatalen)
  except:
    echo("headerdata setPosition error")
    return false
  true

{.pop.}


template handleError() =
  if handler.currentexceptionmsg != "":
    {.gcsafe.}:
      if httpErrorHandlerCallbacks[gs.serverid] != nil: httpErrorHandlerCallbacks[gs.serverid](handler)

proc handleHeader(gs: ptr GuildenServer, data: ptr SocketData) {.gcsafe, nimcall, raises: [].} =
  var finished = false
  var handler = getHeaderSlurper()
  try:
    handler.gs = gs[]
    handler.socket = data.socket
    handler.dataobject = data.dataobject
    if handler.slurpHeader():
      {.gcsafe.}: httpRequestHandlerCallbacks[gs.serverid](handler)
    else:
      finished = true
      handleError()
  except:
    handleError()
    finished = true
  finally:
    if finished: closeFd(handler.gs, handler.socket)   
    handler.ishandling = false
    

proc initHeaderSlurping*(headerslurpingtype: HandlerType, gs: GuildenServer, onrequestcallback: proc(handler: HeaderSlurper){.gcsafe, nimcall, raises: [].}) =
  HeaderSlurping = headerslurpingtype
  initLock(lock)
  {.gcsafe.}: 
    httpRequestHandlerCallbacks[gs.serverid] = onrequestcallback
    gs.registerDefaultHandler(HeaderSlurping, handleHeader)


proc registerSlurperErrorHandler*(gs: GuildenServer, onerrorcallback: proc(handler: Handler){.gcsafe, nimcall, raises: [].}) =
  {.gcsafe.}: httpErrorHandlerCallbacks[gs.serverid] = onerrorcallback
  

proc write*(h: HeaderSlurper, str: string): bool {.raises: [].} =
  try: 
    if h.senddata.getPosition() + str.len() > MaxSlurperResponseLength: return false
    h.senddata.write(str)
  except: return false
  true