from streams import StringStream, newStringStream, getPosition, setPosition, write
from strutils import toLowerAscii, parseInt
from os import osLastError, osErrorMsg, OSErrorCode, sleep
from posix import recv
import locks
import guildenserver, httpctx


type
  FullCtx* = ref object of HttpCtx

var
  FullCtxType: HandlerType
  lock: Lock
  httpRequestHandlerCallbacks: array[256, proc(ctx: FullCtx) {.gcsafe, raises: [].}]
  ctxpool {.threadvar.}: array[MaxHandlerCount, FullCtx]


template newFullCtx() =
  ctxpool[i] = new FullCtx
  ctxpool[i].handlertype = FullCtxType
  newHttpCtxData()
  

proc getFullCtx(): lent FullCtx {.inline, raises: [].} =
  {.gcsafe.}:
    withLock(lock):
      var i = 0
      result = ctxpool[i]
      while ctxpool[i] == nil or ctxpool[i].ishandling:
        if ctxpool[i] == nil:
          newFullCtx()          
          break
        i += 1
        if i == MaxHandlerCount:
          sleep(20)
          i = 0
      result = ctxpool[i]
      result.ishandling = true
    initHttpCtx()


proc isHeaderreceived(ctx: FullCtx, previouslen, currentlen: int): bool =
  var i = if previouslen > 4: previouslen - 4 else: previouslen
  while i <= currentlen - 4:
    if ctx.recvdata.data[i] == '\c' and ctx.recvdata.data[i+1] == '\l' and ctx.recvdata.data[i+2] == '\c' and ctx.recvdata.data[i+3] == '\l':
      ctx.bodystart = i + 4
      return true
    inc i
  false


proc recvHttp(ctx: FullCtx): bool {.gcsafe, raises:[] .} =
  var recvdatalen = 0
  var trials = 0
  var expectedlength = MaxRequestLength + 1
  while true:
    let ret = recv(ctx.socket, addr ctx.recvdata.data[recvdatalen], expectedlength - recvdatalen, 0.cint)
    if ctx.gs.serverstate == Shuttingdown: return false
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
      ctx.currentexceptionmsg = "recvHttp: " & $lastError & " " & osErrorMsg(OSErrorCode(lastError))
      return false
    if trials >= 4:
      ctx.currentexceptionmsg = "recvHttp: receiving stalled"
      return false
      
    let previouslen = recvdatalen
    recvdatalen += ret

    if recvdatalen == MaxRequestLength + 1:
      ctx.currentexceptionmsg = "recvHttp: Max request size exceeded"
      return false
    if recvdatalen == expectedlength: break
    if expectedlength >= MaxRequestLength + 1:
      try:
        if not ctx.isHeaderreceived(previouslen, recvdatalen):
          trials += 1
          continue
        trials = 0
        let contlenpos = ctx.getHeader("content-length")
        if contlenpos == "": return true
        expectedlength = parseInt(contlenpos) + ctx.bodystart
        if recvdatalen == expectedlength: break
      except:
        ctx.currentexceptionmsg = "recvHttp: " & getCurrentExceptionMsg()
        return false
  try:
    ctx.recvdata.setPosition(recvdatalen)
  except:
    echo("recvdata setPosition error")
    return false
  true


proc handleHttp(gs: ptr GuildenServer, data: ptr SocketData) {.gcsafe, nimcall, raises: [].} =
  var finished = false
  var ctx: FullCtx = getFullCtx()
  try:
    ctx.gs = gs[]
    ctx.socket = data.socket
    ctx.dataobject = data.dataobject
    if ctx.recvHttp():
      {.gcsafe.}: httpRequestHandlerCallbacks[gs.serverid](ctx)
    else: finished = true
  except: finished = true
  finally:
    if finished: closeFd(ctx.gs, ctx.socket)   
    ctx.ishandling = false
  if not finished:
    ctx.parseRequestLine()
    ctx.parseHeaders()
    

proc initFullCtx*(ctxtypeid: HandlerType, gs: GuildenServer, onrequestcallback: proc(ctx: FullCtx){.gcsafe, nimcall, raises: [].}) =
  FullCtxType = ctxtypeid
  initLock(lock)
  {.gcsafe.}: 
    httpRequestHandlerCallbacks[gs.serverid] = onrequestcallback
    gs.registerHandler(FullCtxType, handleHttp)