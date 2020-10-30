from selectors import Selector, contains, registerTimer, unregister
from nativesockets import SocketHandle, close
export SocketHandle
from os import getCurrentProcessId
from posix import kill, SIGINT
import locks


const
  RcvTimeOut* {.intdefine.} = 5 # SO_RCVTIMEO, https://linux.die.net/man/7/socket
  
when compileOption("threads"):
  import threadpool
  const MaxHandlerCount* {.intdefine.} = MaxThreadPoolSize
else:
  const MaxHandlerCount* = 1

type
  CtxId* = distinct int

  RequestCallback* = proc(ctx: Ctx) {.gcsafe, raises: [].}
  HandlerAssociation* = tuple[port: int, ctxid: CtxId]
  
  ServerState* = enum
    Initializing, Normal, Maintenance, Shuttingdown

  SocketData* = object
    port*: uint16
    socket*: SocketHandle
    ctxid*: CtxId
    customdata*: pointer

  Ctx* {.inheritable, shallow.} = ref object
    gs*: ptr GuildenServer
    socketdata*: ptr SocketData

  HandlerCallback* = proc(gs: ptr GuildenServer, data: ptr SocketData){.gcsafe, nimcall, raises: [].}
  ErrorCallback* = proc(msg: string) {.gcsafe, raises: [].}

  GuildenServer* {.inheritable.} = ref object
    serverid*: int
    multithreading*: bool
    serverstate*: ServerState
    selector*: Selector[SocketData]
    lock*: Lock
    porthandlers*: array[200, HandlerAssociation]
    portcount*: int    
    handlercallbacks*: array[0.CtxId .. 200.CtxId, HandlerCallback]
    errorNotifier*: ErrorCallback
    shutdownHandler*: proc() {.gcsafe, raises: [].}
    nextctxid: int


const
  InvalidCtx* = 0.CtxId
  ServerCtx* = (-1).CtxId
  TimerCtx* = (-2).CtxId
  SignalCtx* = (-3).CtxId

proc `$`*(x: SocketHandle): string {.inline.} = $(x.cint)
proc `$`*(x: CtxId): string {.inline.} = $(x.int)
proc `==`*(x, y: CtxId): bool {.borrow.}


var serveridlock: Lock
initLock(serveridlock)

var nextserverid: int
proc initGuildenServer*(gs: var GuildenServer) {.gcsafe, nimcall.} = # , handlers: openarray[HandlerAssociation]
  initLock(gs.lock)
  #gs.portcount = handlers.len
  gs.nextctxid = 1
  #for i in 0 ..< gs.portcount:
  #  gs.porthandlers[i] = handlers[i]
  withLock(serveridlock):
    gs.serverid = nextserverid
    nextserverid += 1

proc newGuildenServer*(): GuildenServer =
  result = new GuildenServer
  result.initGuildenServer()


proc getCtxId*(gs: var GuildenServer): CtxId {.gcsafe, nimcall.} =
  withLock(gs.lock):
    result = gs.nextctxid.CtxId
    gs.nextctxid += 1


proc signalSIGINT*() =
  discard kill(getCurrentProcessId().cint, SIGINT)


#proc registerDefaultHandler*(gs: GuildenServer,  ctxid: CtxId, callback: HandlerCallback) =
#  gs.handlercallbacks[ctxid] = callback



proc registerHandler*(gs: GuildenServer,  ctxid: CtxId, callback: HandlerCallback, ports: openArray[int]) =
  assert(ctxid.int > 0, "ctx types below 1 are reserved for internal use")
  gs.handlercallbacks[ctxid] = callback
  for port in ports:
    gs.portHandlers[gs.portcount] = (port, ctxid)
    gs.portcount += 1



#proc registerTimerhandler*(gs: GuildenServer, callback: proc() {.gcsafe, nimcall, raises: [].}, interval: int) =
#  gs.timerHandler = callback
#  discard gs.selector.registerTimer(interval, false, SocketData(fdKind: Ticker, customdata: nil))



proc registerErrornotifier*(gs: GuildenServer, callback: ErrorCallback) =
  gs.errorNotifier = callback


proc registerShutdownhandler*(gs: GuildenServer, callback: proc() {.gcsafe, nimcall, raises: [].}) =
  gs.shutdownHandler = callback


proc handleRead*(gs: ptr GuildenServer, data: ptr SocketData) =
  assert(gs.handlercallbacks[data.ctxid] != nil, "No ctx registered for CtxId " & $data.ctxid)
  gs.handlercallbacks[data.ctxid](gs, data)


proc closeSocket*(ctx: Ctx) {.raises: [].} =
  withLock(ctx.gs.lock):
    try:
      if ctx.gs.selector.contains(ctx.socketdata.socket): ctx.gs.selector.unregister(ctx.socketdata.socket)
      ctx.socketdata.socket.close()
    except: discard