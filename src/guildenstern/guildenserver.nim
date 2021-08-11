from selectors import Selector, newselector, contains, registerTimer, unregister, getData, newSelectEvent, registerEvent, trigger
from posix import SocketHandle, INVALID_SOCKET, SIGINT, getpid, SIGTERM, onSignal, `==`
from nativesockets import close
export SocketHandle, INVALID_SOCKET, posix.`==`


const
  MaxCtxHandlers* {.intdefine.} = 100
  RcvTimeOut* {.intdefine.} = 5 # SO_RCVTIMEO, https://linux.die.net/man/7/socket

type
  CtxId* = distinct int

  RequestCallback* = proc(ctx: Ctx) {.nimcall, raises: [].}

  HandlerCallback* = proc(gs: ptr GuildenServer, data: ptr SocketData){.nimcall, gcsafe, raises: [].}

  HandlerAssociation* = tuple[port: uint16, protocol: int, handler: HandlerCallback]

  SocketData* = object
    port*: uint16
    socket*: posix.SocketHandle
    ctxid*: CtxId
    customdata*: pointer
    flags*: int

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

  TimerCallback* = proc() {.nimcall, raises: [].}
  CloseCallback* = proc(ctx: Ctx, socket: SocketHandle, cause: SocketCloseCause, msg: string){.gcsafe, nimcall, raises: [].}
  
  GuildenServer* {.inheritable.} = ref object
    multithreading*: bool
    selector*: Selector[SocketData]
    porthandlers*: array[0.CtxId .. MaxCtxHandlers.CtxId, HandlerAssociation]
    closecallback*: CloseCallback
    nextctxid: int
    protocolnames: seq[string]


const
  InvalidCtx* = 0.CtxId
  ServerCtx* = (-1).CtxId
  TimerCtx* = (-2).CtxId

proc `$`*(x: SocketHandle): string {.inline.} = $(x.cint)
proc `$`*(x: CtxId): string {.inline.} = $(x.int)
proc `==`*(x, y: CtxId): bool {.borrow.}


var shuttingdown* = false
var shutdownevent = newSelectEvent()

proc shutdown*() =
  {.gcsafe.}: shuttingdown = true
  try: trigger(shutdownevent)
  except:
    echo getCurrentExceptionMsg()
    quit(-666)

onSignal(SIGTERM): shutdown()
onSignal(SIGINT): shutdown()


proc initialize(gs: var GuildenServer) =
  gs.nextctxid = 1
  gs.selector = newSelector[SocketData]()
  gs.selector.registerEvent(shutdownevent, SocketData())
  gs.protocolnames = @["unknown"]


proc getProtocolindex(gs: GuildenServer, protocolname: string): int =
  result = gs.protocolnames.find(protocolname)
  if result == -1:
    gs.protocolnames.add(protocolname)
    result = gs.protocolnames.len - 1


proc generateCtxId(gs: var GuildenServer): CtxId =
  assert(gs.nextctxid < MaxCtxHandlers, "Cannot create more handlers")
  result = gs.nextctxid.CtxId
  gs.nextctxid += 1


proc registerHandler*(gs: var GuildenServer, callback: HandlerCallback, port: int, protocolname: string) =
  if gs.nextctxid == 0: gs.initialize()
  let protocol = gs.getProtocolindex(protocolname)
  let ctxid = gs.generateCtxId()
  gs.portHandlers[ctxid] = (port.uint16, protocol, callback)


proc getProtocolName*(ctx: Ctx): string =
  ctx.gs.protocolnames[ctx.gs.portHandlers[ctx.socketdata.ctxid].protocol]


proc registerTimerhandler*(gs: var GuildenServer, callback: TimerCallback, interval: int) =
  if gs.nextctxid == 0: gs.initialize()
  discard gs.selector.registerTimer(interval, false, SocketData(ctxid: TimerCtx, customdata: cast[pointer](callback)))


proc registerConnectionclosedhandler*(gs: GuildenServer, callback: CloseCallback) =
  gs.closecallback = callback


proc handleRead*(gs: ptr GuildenServer, data: ptr SocketData) =
  assert(gs.porthandlers[data.ctxid].handler != nil, "No ctx registered for CtxId " & $data.ctxid)
  {.gcsafe.}: gs.porthandlers[data.ctxid].handler(gs, data)


{.push hints:off.}
proc closeSocket*(ctx: Ctx, cause = CloseCalled, msg = "") {.raises: [].} =
  if ctx.socketdata.socket.int in [0, INVALID_SOCKET.int]: return
  when defined(fulldebug): echo "socket ", cause, ": ", ctx.socketdata.socket, "  ", msg
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
  when defined(fulldebug): echo "closeOtherSocket ", cause, ": ", data.socket, "  ", msg
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
{.pop.}


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