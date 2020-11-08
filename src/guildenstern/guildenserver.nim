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
    socket*: SocketHandle
    ctxid*: CtxId
    customdata*: pointer

  Ctx* {.inheritable, shallow.} = ref object
    gs*: ptr GuildenServer
    socketdata*: ptr SocketData

  HandlerCallback* = proc(gs: ptr GuildenServer, data: ptr SocketData){.gcsafe, nimcall, raises: [].}
  TimerCallback* = proc() {.nimcall, gcsafe, raises: [].}
  ThreadInitializationCallback* = proc() {.nimcall, gcsafe, raises: [].}
  ErrorCallback* = proc(msg: string) {.gcsafe, raises: [].}
  
  GuildenServer* {.inheritable.} = ref object
    multithreading*: bool
    selector*: Selector[SocketData]
    lock*: Lock
    porthandlers*: array[MaxCtxHandlers, HandlerAssociation]
    portcount*: int
    threadinitializer*: ThreadInitializationCallback   
    handlercallbacks*: array[0.CtxId .. MaxCtxHandlers.CtxId, HandlerCallback]
    errorNotifier*: ErrorCallback
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

proc initGuildenServer*(gs: var GuildenServer) {.gcsafe, nimcall.} =
  initLock(gs.lock)
  gs.selector = newSelector[SocketData]()
  gs.nextctxid = 1

proc newGuildenServer*(): GuildenServer =
  result = new GuildenServer
  result.initGuildenServer()


proc shutdown*() {.gcsafe, noconv.} =
  {.gcsafe.}: shuttingdown = true

setControlCHook(shutdown)


proc getCtxId*(gs: var GuildenServer): CtxId {.gcsafe, nimcall.} =
  withLock(gs.lock):
    assert(gs.nextctxid < MaxCtxHandlers, "Cannot create more handlers")    
    result = gs.nextctxid.CtxId
    gs.nextctxid += 1


proc notifyError*(ctx: Ctx, msg: string) {.inline.} =
  if ctx.gs.errorNotifier != nil: ctx.gs.errorNotifier(msg)
  else:
    if defined(fulldebug): echo msg


proc registerThreadInitializer*(gs: GuildenServer, callback: ThreadInitializationCallback) =
  gs.threadinitializer = callback


proc registerHandler*(gs: GuildenServer,  ctxid: CtxId, callback: HandlerCallback, ports: openArray[int]) =
  assert(ctxid.int > 0, "ctx types below 1 are reserved for internal use")
  assert(ctxid.int < MaxCtxHandlers, "ctxid " & $ctxid & "over MaxCtxHandlers = " & $MaxCtxHandlers)
  assert(gs.nextctxid > 0, "Ctx handlers can be registered only after initGuildenServer is called")
  gs.handlercallbacks[ctxid] = callback
  for port in ports:
    gs.portHandlers[gs.portcount] = (port, ctxid)
    gs.portcount += 1


proc registerTimerhandler*(gs: GuildenServer, callback: TimerCallback, interval: int) =
  assert(gs.nextctxid > 0, "Timer handlers can be registered only after initGuildenServer is called")
  discard gs.selector.registerTimer(interval, false, SocketData(ctxid: TimerCtx, customdata: cast[pointer](callback)))


proc registerErrornotifier*(gs: GuildenServer, callback: ErrorCallback) =
  gs.errorNotifier = callback


proc handleRead*(gs: ptr GuildenServer, data: ptr SocketData) =
  assert(gs.handlercallbacks[data.ctxid] != nil, "No ctx registered for CtxId " & $data.ctxid)
  gs.handlercallbacks[data.ctxid](gs, data)


proc closeSocket*(ctx: Ctx) {.raises: [].} =
  try:
    withLock(ctx.gs.lock):
      when defined(fulldebug): echo "closing ctx socket: ", ctx.socketdata.socket
      ctx.socketdata.socket.close()
      if ctx.gs.selector.contains(ctx.socketdata.socket): ctx.gs.selector.unregister(ctx.socketdata.socket)       
      ctx.socketdata.socket = osInvalidSocket
  except: discard


proc closeSocket*(gs: ptr GuildenServer, socket: SocketHandle) {.raises: [].} =
  try:
    withLock(gs.lock):
      when defined(fulldebug): echo "closing socket: ", socket
      socket.close()
      if gs.selector.contains(socket): gs.selector.unregister(socket)
  except: discard