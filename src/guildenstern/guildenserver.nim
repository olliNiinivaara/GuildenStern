from selectors import Selector, newselector, contains, registerTimer, unregister
from nativesockets import SocketHandle, osInvalidSocket, close
export SocketHandle
import locks


const
  MaxCtxHandlers* {.intdefine.} = 100
  RcvTimeOut* {.intdefine.} = 5 # SO_RCVTIMEO, https://linux.die.net/man/7/socket
  
type
  CtxId* = distinct int

  RequestCallback* = proc(ctx: Ctx) {.gcsafe, raises: [].}
  HandlerAssociation* = tuple[port: int, ctxid: CtxId]
  
  SocketData* = object
    port*: uint16
    socket*: nativesockets.SocketHandle
    ctxid*: CtxId
    customdata*: pointer

  Ctx* {.inheritable, shallow.} = ref object
    gs*: ptr GuildenServer
    socketdata*: ptr SocketData

  HandlerCallback* = proc(gs: ptr GuildenServer, data: ptr SocketData){.gcsafe, nimcall, raises: [].}
  TimerCallback* = proc() {.nimcall, gcsafe, raises: [].}
  ThreadInitializationCallback* = proc() {.nimcall, gcsafe, raises: [].}
  LostCallback* = proc(gs: ptr GuildenServer, data: ptr SocketData, closedsocket: SocketHandle){.gcsafe, nimcall, raises: [].}
  ErrorCallback* = proc(msg: string) {.gcsafe, raises: [].}
  
  GuildenServer* {.inheritable.} = ref object
    multithreading*: bool
    selector*: Selector[SocketData]
    lock*: Lock
    porthandlers*: array[MaxCtxHandlers, HandlerAssociation]
    portcount*: int
    threadinitializer*: ThreadInitializationCallback   
    handlercallbacks*: array[0.CtxId .. MaxCtxHandlers.CtxId, HandlerCallback]
    errornotifier*: ErrorCallback
    lostcallback*: LostCallback
    nextctxid: int


const
  InvalidCtx* = 0.CtxId
  ServerCtx* = (-1).CtxId
  TimerCtx* = (-2).CtxId
  SignalCtx* = (-3).CtxId

proc `$`*(x: SocketHandle): string {.inline.} = $(x.cint)
proc `$`*(x: CtxId): string {.inline.} = $(x.int)
proc `==`*(x, y: CtxId): bool {.borrow.}

var shuttingdown* = false

proc shutdown*() {.gcsafe, noconv.} =
  {.gcsafe.}: shuttingdown = true

setControlCHook(shutdown)


proc getCtxId(gs: var GuildenServer): CtxId {.gcsafe, nimcall.} =
  if gs.nextctxid == 0:
    gs.nextctxid = 1
    gs.selector = newSelector[SocketData]()
  assert(gs.nextctxid < MaxCtxHandlers, "Cannot create more handlers")    
  result = gs.nextctxid.CtxId
  gs.nextctxid += 1


proc notifyError*(gs: ptr GuildenServer, msg: string) {.inline.} =
  if gs.errorNotifier != nil: gs.errorNotifier(msg)
  else:
    if defined(fulldebug): echo msg


proc registerThreadInitializer*(gs: GuildenServer, callback: ThreadInitializationCallback) =
  gs.threadinitializer = callback


proc registerHandler*(gs: var GuildenServer, callback: HandlerCallback, port: int): CtxId =
  result = gs.getCtxId()
  gs.handlercallbacks[result] = callback
  if port > 0:
    gs.portHandlers[gs.portcount] = (port, result)
    gs.portcount += 1


proc registerTimerhandler*(gs: GuildenServer, callback: TimerCallback, interval: int) =
  discard gs.selector.registerTimer(interval, false, SocketData(ctxid: TimerCtx, customdata: cast[pointer](callback)))


proc registerConnectionlosthandler*(gs: GuildenServer, callback: LostCallback) =
  gs.lostcallback = callback


proc registerErrornotifier*(gs: GuildenServer, callback: ErrorCallback) =
  gs.errorNotifier = callback


proc handleRead*(gs: ptr GuildenServer, data: ptr SocketData) =
  assert(gs.handlercallbacks[data.ctxid] != nil, "No ctx registered for CtxId " & $data.ctxid)
  gs.handlercallbacks[data.ctxid](gs, data)


proc closeSocket*(ctx: Ctx) {.raises: [].} =
  try:
    when defined(fulldebug): echo "closing ctx socket: ", ctx.socketdata.socket
    if ctx.gs.selector.contains(ctx.socketdata.socket): ctx.gs.selector.unregister(ctx.socketdata.socket)
    # finally got it?
    ctx.socketdata.socket.close()
    ctx.socketdata.socket = osInvalidSocket
  except:
    if defined(fulldebug): echo "close error: ", getCurrentExceptionMsg()


proc closeSocket*(gs: ptr GuildenServer, socket: SocketHandle) {.raises: [].} =
  try:
    when defined(fulldebug): echo "closing socket: ", socket
    if gs.selector.contains(socket): gs.selector.unregister(socket)
    socket.close()    
  except:
    if defined(fulldebug): echo "close error: ", getCurrentExceptionMsg()


proc  handleConnectionlost*(gs: ptr GuildenServer, data: ptr SocketData, lostsocket: SocketHandle) {.raises: [].} =
  if gs.lostcallback != nil: gs.lostcallback(gs, data, lostsocket)