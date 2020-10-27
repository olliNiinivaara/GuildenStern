from streams import StringStream, newStringStream, getPosition, setPosition, write
from os import osLastError, osErrorMsg, OSErrorCode, sleep
from posix import recv
import locks
import guildenserver
import httpctx
export httpctx

type
  HeaderCtx* = ref object of HttpCtx
 
var
  HeaderCtxType: HandlerType
  lock: Lock
  httpRequestHandlerCallbacks: array[256, proc(ctx: HeaderCtx) {.gcsafe, raises: [].}]
  ctxpool {.threadvar.}: array[MaxHandlerCount, HeaderCtx]


template newHeaderCtx() =
  ctxpool[i] = new HeaderCtx
  ctxpool[i].handlertype = HeaderCtxType
  newHttpCtxData()
  

proc getHeaderCtx(): lent HeaderCtx {.inline, raises: [].} =
 {.gcsafe.}:
    withLock(lock):
      var i = 0
      result = ctxpool[i]
      while ctxpool[i] == nil or ctxpool[i].ishandling:
        if ctxpool[i] == nil:
          newHeaderCtx()          
          break
        i += 1
        if i == MaxHandlerCount:
          sleep(20)
          i = 0
      result = ctxpool[i]
      result.ishandling = true
    initHttpCtx()


{.push checks: off.}

template isFinished: bool =
  ctx.recvdata.data[recvdatalen-4] == '\c' and ctx.recvdata.data[recvdatalen-3] == '\l' and ctx.recvdata.data[recvdatalen-2] == '\c' and ctx.recvdata.data[recvdatalen-1] == '\l'
  
proc receiveHeader*(ctx: HeaderCtx): bool {.gcsafe, raises:[].} =
  var recvdatalen = 0
  var trials = 0
  while true:
    if ctx.gs.serverstate == Shuttingdown: return false
    let ret = recv(ctx.socket, addr ctx.recvdata.data[recvdatalen], MaxRequestLength + 1, 0.cint)
    if ctx.gs.serverstate == Shuttingdown: return false
    echo ret
    if ret == 0: return false
    #[  let lastError = osLastError().int
      if lastError == 2 or lastError == 9 or lastError == 104: return false
      if lastError != 0:
        echo "lasteroor ", lastError
        trials.inc
        if trials <= 3: 
          sleep(100 + trials * 100)
          continue
        else:
          ctx.currentexceptionmsg = "receiveHeader: receiving stalled"
          return false
      else: break]#
    if ret == -1:
      let lastError = osLastError().int
      ctx.currentexceptionmsg = "receiveHeader: " & $lastError & " " & osErrorMsg(OSErrorCode(lastError))
      return false      
    recvdatalen += ret
    if recvdatalen >= MaxRequestLength + 1:
      ctx.currentexceptionmsg = "receiveHeader: Max request size exceeded"
      return false
    if isFinished:
      break
    else:
      echo "ootetaan lisää"
      echo ctx.recvdata.data[0 .. recvdatalen]
  try:
    ctx.recvdata.setPosition(recvdatalen)
    ctx.requestlen = recvdatalen
  except:
    echo("recvdata setPosition error")
    return false
  return ctx.requestlen > 0
  
{.pop.}


proc handleHeaderRequest(gs: ptr GuildenServer, data: ptr SocketData) {.gcsafe, nimcall, raises: [].} =
  var finished = false
  var ctx = getHeaderCtx()
  try:
    ctx.gs = gs
    ctx.socket = data.socket
    ctx.dataobject = data.dataobject
    if ctx.receiveHeader():
      ctx.parseRequestLine() 
      {.gcsafe.}: httpRequestHandlerCallbacks[gs.serverid](ctx)
    else:
      discard ctx.handleError()
      finished = true 
  except: finished = true
  finally:
    if finished: closeFd(ctx.gs, ctx.socket)
    ctx.ishandling = false
    

proc initHeaderCtx*(headerctxtypeid: HandlerType, gs: GuildenServer, onrequestcallback: proc(ctx: HeaderCtx){.gcsafe, nimcall, raises: [].}) =
  HeaderCtxType = headerctxtypeid
  initLock(lock)
  {.gcsafe.}: 
    httpRequestHandlerCallbacks[gs.serverid] = onrequestcallback
    gs.registerHandler(HeaderCtxType, handleHeaderRequest)