from selectors import Selector, newselector, contains, registerTimer, unregister, getData
from posix import SocketHandle, INVALID_SOCKET, SIGINT, getpid
from posix_utils import sendSignal
from nativesockets import close
export SocketHandle, INVALID_SOCKET


const
  MaxCtxHandlers* {.intdefine.} = 100
  RcvTimeOut* {.intdefine.} = 5 # SO_RCVTIMEO, https://linux.die.net/man/7/socket

type
  CtxId* = distinct int

  RequestCallback* = proc(ctx: Ctx) {.nimcall, raises: [].}
  HandlerAssociation* = tuple[port: int, ctxid: CtxId, protocol: int]

  SocketData* = object
    port*: uint16
    socket*: posix.SocketHandle
    ctxid*: CtxId
    protocol*: int
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
    SecurityThreatened
    DontClose

  HandlerCallback* = proc(gs: ptr GuildenServer, data: ptr SocketData){.nimcall, raises: [].}
  TimerCallback* = proc() {.nimcall, raises: [].}
  ThreadInitializationCallback* = proc() {.nimcall, gcsafe, raises: [].}
  CloseCallback* = proc(ctx: Ctx, socket: SocketHandle, cause: SocketCloseCause, msg: string){.gcsafe, nimcall, raises: [].}
  
  GuildenServer* {.inheritable.} = ref object
    multithreading*: bool
    selector*: Selector[SocketData]
    porthandlers*: array[MaxCtxHandlers, HandlerAssociation]
    portcount*: int
    threadinitializer*: ThreadInitializationCallback
    handlercallbacks*: array[0.CtxId .. MaxCtxHandlers.CtxId, HandlerCallback]
    closecallback*: CloseCallback
    nextctxid: int
    protocolnames: seq[string]


const
  InvalidCtx* = 0.CtxId
  ServerCtx* = (-1).CtxId
  TimerCtx* = (-2).CtxId
  SignalCtx* = (-3).CtxId

proc `$`*(x: SocketHandle): string {.inline.} = $(x.cint)
proc `$`*(x: CtxId): string {.inline.} = $(x.int)
proc `==`*(x, y: CtxId): bool {.borrow.}

var shuttingdown* = false

proc shutdown*() =
  {.gcsafe.}: shuttingdown = true
  try: sendSignal(getpid(), SIGINT)
  except: discard

proc doShutdown() {.gcsafe, noconv.} =
  shutdown()

setControlCHook(doShutdown)


proc getCtxId(gs: var GuildenServer, protocolname: string): (CtxId, int) {.gcsafe, nimcall.} =
  if gs.nextctxid == 0:
    gs.nextctxid = 1
    gs.selector = newSelector[SocketData]()
    gs.protocolnames = @["unknown"]
  assert(gs.nextctxid < MaxCtxHandlers, "Cannot create more handlers")
  var protocol = gs.protocolnames.find(protocolname)
  if protocol == -1:
    gs.protocolnames.add(protocolname)
    protocol = gs.protocolnames.len
  result = (gs.nextctxid.CtxId, protocol)
  gs.nextctxid += 1


proc registerThreadInitializer*(gs: GuildenServer, callback: ThreadInitializationCallback) =
  gs.threadinitializer = callback


proc registerHandler*(gs: var GuildenServer, callback: HandlerCallback, port: int, protocolname: string): CtxId =
  var protocol: int
  (result , protocol) = gs.getCtxId(protocolname)
  gs.handlercallbacks[result] = callback
  if port > 0:
    gs.portHandlers[gs.portcount] = (port, result, protocol)
    gs.portcount += 1


proc registerTimerhandler*(gs: GuildenServer, callback: TimerCallback, interval: int) =
  if gs.nextctxid == 0:
    gs.nextctxid = 1
    gs.selector = newSelector[SocketData]()
  discard gs.selector.registerTimer(interval, false, SocketData(ctxid: TimerCtx, customdata: cast[pointer](callback)))


proc registerConnectionclosedhandler*(gs: GuildenServer, callback: CloseCallback) =
  gs.closecallback = callback


proc getProtocolName*(ctx: Ctx): string =
  return ctx.gs.protocolnames[ctx.socketdata.protocol]


proc handleRead*(gs: ptr GuildenServer, data: ptr SocketData) =
  assert(gs.handlercallbacks[data.ctxid] != nil, "No ctx registered for CtxId " & $data.ctxid)
  {.gcsafe.}: gs.handlercallbacks[data.ctxid](gs, data)


proc closeSocket*(ctx: Ctx, cause = CloseCalled, msg = "") {.raises: [].} =
  if ctx.socketdata.socket.int in [0, INVALID_SOCKET.int]: return
  when defined(fulldebug): echo "socket ", cause, ": ", ctx.socketdata.socket
  let fd = ctx.socketdata.socket
  try:
    if ctx.gs.closecallback != nil: ctx.gs.closecallback(ctx, fd, cause, msg)
    if ctx.gs.selector.contains(fd.int):
      ctx.gs.selector.unregister(fd.int)
    if cause notin [ClosedbyClient, ConnectionLost]: nativesockets.SocketHandle(fd).close()
    ctx.socketdata.socket = INVALID_SOCKET
  except:
    when defined(fulldebug): echo "close error: ", getCurrentExceptionMsg()


proc closeOtherSocket*(gs: ptr GuildenServer, data: ptr SocketData, cause: SocketCloseCause, msg: string) {.raises: [].} =
  if data == nil or data.socket.int in [0, INVALID_SOCKET.int]: return
  when defined(fulldebug): echo "closeOtherSocket ", cause, ": ", data.socket
  try:
    let fd = data.socket
    data.socket = INVALID_SOCKET
    if gs.closecallback != nil:      
      var ctx = new Ctx
      ctx.gs = gs
      ctx.socketdata = data      
      gs.closecallback(ctx, fd, cause, msg) 
    if gs.selector.contains(fd.int): gs.selector.unregister(fd.int)
    if cause notin [ClosedbyClient, ConnectionLost]: nativesockets.SocketHandle(fd).close()
  except:
    when defined(fulldebug): echo "internal close error: ", getCurrentExceptionMsg()


proc closeOtherSocket*(gs: GuildenServer, socket: posix.SocketHandle, cause: SocketCloseCause, msg: string = "") {.raises: [].} =
  if socket.int in [0, INVALID_SOCKET.int]: return
  var data: ptr SocketData
  try:
    {.push warning[ProveInit]: off.}
    data = addr(gs.selector.getData(socket.int))
    {.pop.}
  except: return
  if data == nil:  return 
  closeOtherSocket(unsafeAddr gs, data, cause, msg)