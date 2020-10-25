from streams import StringStream, newStringStream, getPosition, setPosition, write
from os import sleep
import locks
import guildenserver, private/[httphandlertype, httpin, httpout]
export HttpHandler
export reply, replyHeaders, replyCode


const
  MaxHttpHeaderValueLength* {.intdefine.} = 200

var
  lock: Lock
  httpRequestHandlerCallbacks: array[256, proc(handler: HttpHandler) {.gcsafe, raises: [].}] # 256 servers in one executable is enough, right?
  httpErrorHandlerCallbacks: array[256, proc(handler: Handler) {.gcsafe, raises: [].}]
  httphandlers: array[MaxHandlerCount, HttpHandler]


proc newHttpHandler(): HttpHandler =
  result = new HttpHandler
  result.handlertype = HttpHandlerType
  result.senddata = newStringStream()
  result.senddata.data.setLen(MaxResponseLength)
  result.recvdata = newStringStream()
  result.recvdata.data.setLen(MaxResponseLength)
  

proc getHttpHandler(): HttpHandler {.inline, raises: [].} =
  {.gcsafe.}:
    withLock(lock):
      var i = 0
      result = httphandlers[i]
      while result == nil or result.ishandling:
        if result == nil:
          result = newHttpHandler()
          httphandlers[i] = result          
          break
        i += 1
        if i == MaxHandlerCount:
          sleep(20) # backoff...
          i = 0
        result = httphandlers[i]
      result.ishandling = true
    result.currentexceptionmsg.setLen(0)
    result.path = 0
    result.pathlen = 0
    result.methlen = 0
    result.bodystartpos = 0
    try:
      result.senddata.setPosition(0)
      result.recvdata.setPosition(0)
    except: (echo "Nim internal error"; return)


template handleError() =
  if handler.currentexceptionmsg != "":
    {.gcsafe.}:
      if httpErrorHandlerCallbacks[gs.serverid] != nil: httpErrorHandlerCallbacks[gs.serverid](handler)

proc handleHttp(gs: ptr GuildenServer, data: ptr SocketData) {.gcsafe, nimcall, raises: [].} =
  var finished = false
  var handler = getHttpHandler()
  try:
    handler.gs = gs[]
    handler.socket = data.socket
    handler.dataobject = data.dataobject
    if handler.readFromHttp():
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
    

proc initHttpHandling*(gs: GuildenServer, onrequestcallback: proc(handler: HttpHandler){.gcsafe, nimcall, raises: [].}, interestingheaderfields: openarray[string] = ["content-length"]) =
  initLock(lock)
  {.gcsafe.}: 
    httpRequestHandlerCallbacks[gs.serverid] = onrequestcallback
    gs.registerDefaultHandler(HttpHandlerType, handleHttp)
    for i in 0 .. interestingheaderfields.high: headerfieldsArrays[gs.serverid][i] = interestingheaderfields[i]
    lastheaders[gs.serverid] = interestingheaderfields.high


proc registerHttpErrorHandler*(gs: GuildenServer, onerrorcallback: proc(handler: Handler){.gcsafe, nimcall, raises: [].}) =
  {.gcsafe.}: httpErrorHandlerCallbacks[gs.serverid] = onerrorcallback
  

proc getPath*(h: HttpHandler): string {.raises: [].} =
  if h.pathlen == 0: return
  return h.recvdata.data[h.path ..< h.path + h.pathlen]

proc isPath*(h: HttpHandler, apath: string): bool {.raises: [].} =
  if h.pathlen != apath.len: return false
  for i in 0 ..< h.pathlen:
    if h.recvdata.data[h.path + i] != apath[i]: return false
  return true

proc pathStarts*(h: HttpHandler, pathstart: string): bool  {.raises: [].} =
  if h.pathlen < pathstart.len: return false
  for i in 0 ..< pathstart.len:
    if h.recvdata.data[h.path + i] != pathstart[i]: return false
  return true

proc getMethod*(h: HttpHandler): string  {.raises: [].} =
  if h.methlen == 0: return
  return h.recvdata.data[0 ..< h.methlen]

proc isMethod*(h: HttpHandler, amethod: string): bool  {.raises: [].} =
  if h.methlen != amethod.len: return false
  for i in 0 ..< h.methlen:
    if h.recvdata.data[i] != amethod[i]: return false
  return true

proc getHeader*(h: HttpHandler, header: string): string {.inline, raises: [].} =
  let tokeni = headerfieldsArrays[h.gs.serverid].find(header)
  if tokeni == -1: return
  return h.headervalues[tokeni]

proc getHeaders*(h: HttpHandler): string  {.raises: [].} =
  try: result = h.recvdata.data[0 ..< h.bodystartpos]
  except: doaSsert(false, "getHeaders error: " & getCurrentExceptionMsg())
  
proc getBody*(h: HttpHandler): string {.raises: [].} =
  try: result = h.recvdata.data[h.bodystartpos ..< h.recvdata.getPosition()]
  except: doaSsert(false, "getBody error: " & getCurrentExceptionMsg())

proc isBody*(h: HttpHandler, body: string): bool {.raises: [].} =
  let len =
    try: h.recvdata.getPosition() - h.bodystartpos
    except: -1
  if  len != body.len: return false
  for i in h.bodystartpos ..< h.bodystartpos + len:
    if h.recvdata.data[i] != body[i]: return false
  return true

proc write*(h: HttpHandler, str: string): bool {.raises: [].} =
  try: 
    if h.senddata.getPosition() + str.len() > MaxResponseLength: return false
    h.senddata.write(str)
  except:  return false
  true