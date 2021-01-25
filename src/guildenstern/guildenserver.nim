from selectors import Selector, newselector, contains, registerTimer, unregister
from nativesockets import SocketHandle, osInvalidSocket, close
export SocketHandle
import locks


const
  MaxCtxHandlers* {.intdefine.} = 100
  RcvTimeOut* {.intdefine.} = 5 # SO_RCVTIMEO, https://linux.die.net/man/7/socket

type
  CtxId* = distinct int

  RequestCallback* = proc(ctx: Ctx) {.nimcall, raises: [].}
  HandlerAssociation* = tuple[port: int, ctxid: CtxId]

  SocketData* = object
    port*: uint16
    socket*: nativesockets.SocketHandle
    ctxid*: CtxId
    customdata*: pointer

  Ctx* {.inheritable, shallow.} = ref object
    gs*: ptr GuildenServer
    socketdata*: ptr SocketData

  SocketCloseCause* = enum
    CloseCalled
    AlreadyClosed
    ClosedbyClient
    ConnectionLost
    TimedOut
    ProtocolViolated
    NetErrored
    Excepted

  HandlerCallback* = proc(gs: ptr GuildenServer, data: ptr SocketData){.nimcall, raises: [].}
  TimerCallback* = proc() {.nimcall, raises: [].}
  ThreadInitializationCallback* = proc() {.nimcall, gcsafe, raises: [].}
  CloseCallback* = proc(ctx: Ctx, cause: SocketCloseCause, msg: string){.gcsafe, nimcall, raises: [].}
  
  GuildenServer* {.inheritable.} = ref object
    multithreading*: bool
    selector*: Selector[SocketData]
    lock*: Lock
    porthandlers*: array[MaxCtxHandlers, HandlerAssociation]
    portcount*: int
    threadinitializer*: ThreadInitializationCallback
    handlercallbacks*: array[0.CtxId .. MaxCtxHandlers.CtxId, HandlerCallback]
    closecallback*: CloseCallback
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


proc registerThreadInitializer*(gs: GuildenServer, callback: ThreadInitializationCallback) =
  gs.threadinitializer = callback


proc registerHandler*(gs: var GuildenServer, callback: HandlerCallback, port: int): CtxId =
  result = gs.getCtxId()
  gs.handlercallbacks[result] = callback
  if port > 0:
    gs.portHandlers[gs.portcount] = (port, result)
    gs.portcount += 1


proc registerTimerhandler*(gs: GuildenServer, callback: TimerCallback, interval: int) =
  if gs.nextctxid == 0:
    gs.nextctxid = 1
    gs.selector = newSelector[SocketData]()
  discard gs.selector.registerTimer(interval, false, SocketData(ctxid: TimerCtx, customdata: cast[pointer](callback)))


proc registerConnectionclosedhandler*(gs: GuildenServer, callback: CloseCallback) =
  gs.closecallback = callback


proc handleRead*(gs: ptr GuildenServer, data: ptr SocketData) =
  assert(gs.handlercallbacks[data.ctxid] != nil, "No ctx registered for CtxId " & $data.ctxid)
  {.gcsafe.}: gs.handlercallbacks[data.ctxid](gs, data)


proc closeSocket*(ctx: Ctx, cause = CloseCalled, msg = "") {.raises: [].} =
  if ctx.socketdata.socket.int in [0, osInvalidSocket.int]: return
  when defined(fulldebug): echo "socket ", cause, ": ", ctx.socketdata.socket
  # ctx.socketdata is zero'ed during unregister(), so need to save `fd`
  let fd = ctx.socketdata.socket
  try:
    if ctx.gs.closecallback != nil: ctx.gs.closecallback(ctx, cause, msg)
    if ctx.gs.selector.contains(fd): ctx.gs.selector.unregister(fd)
    if cause notin [ClosedbyClient, ConnectionLost]: fd.close()
    ctx.socketdata.socket = osInvalidSocket
  except:
    when defined(fulldebug): echo "close error: ", getCurrentExceptionMsg()


proc internalCloseSocket*(gs: ptr GuildenServer, data: ptr SocketData, cause: SocketCloseCause, msg: string) {.raises: [].} =
  when defined(fulldebug): echo "internally socket ", cause, ": ", data.socket
  try:
    if gs.closecallback != nil:
      var ctx = new Ctx
      ctx.gs = gs
      ctx.socketdata = data
      ctx.gs.closecallback(ctx, cause, msg)
    let fd = data.socket  
    if gs.selector.contains(fd): gs.selector.unregister(fd)
    fd.close()
    data.socket = osInvalidSocket
  except:
    when defined(fulldebug): echo "internal close error: ", getCurrentExceptionMsg()