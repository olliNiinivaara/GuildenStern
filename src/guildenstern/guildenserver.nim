from selectors import Selector, contains, registerTimer, unregister
from nativesockets import SocketHandle, close
from net import Port
from os import getCurrentProcessId
from posix import kill, SIGINT
from streams import StringStream
import locks


const
  RcvTimeOut* {.intdefine.} = 5 # SO_RCVTIMEO, https://linux.die.net/man/7/socket
  
when compileOption("threads"):
  import threadpool
  const MaxHandlerCount* {.intdefine.} = MaxThreadPoolSize
else:
  const MaxHandlerCount* = 1

type
  HandlerType* = distinct int

  HandlerAssociation* = tuple[port: int, handlertype: HandlerType]
  
  ServerState* = enum
    Initializing, Normal, Maintenance, Shuttingdown

  SocketData* = object
    port*: uint16
    socket*: SocketHandle
    handlertype*: HandlerType
    dataobject*: ref RootObj

  HandlerCallback* = proc(gs: ptr GuildenServer, data: ptr SocketData){.gcsafe, nimcall, raises: [].}
  ErrorCallback* = proc(ctx: Ctx) {.gcsafe, raises: [].}

  GuildenServer* {.inheritable.} = ref object
    serverid*: int
    multithreading*: bool
    serverstate*: ServerState
    selector*: Selector[SocketData]
    porthandlers*: array[200, HandlerAssociation]
    portcount*: int    
    handlercallbacks*: array[0.HandlerType .. 200.HandlerType, HandlerCallback]
    errorHandler*: ErrorCallback
    shutdownHandler*: proc() {.gcsafe, raises: [].}

  Ctx* {.inheritable, shallow.} = ref object
    handlertype*: HandlerType
    gs*: ptr GuildenServer
    socket*: SocketHandle
    dataobject*: ref RootObj
    ishandling*: bool
    recvdata* : StringStream
    senddata* : StringStream
    currentexceptionmsg* : string
  


const
  InvalidHandling* = 0.HandlerType
  ServerHandling* = (-1).HandlerType
  TimerHandling* = (-2).HandlerType
  SignalHandling* = (-3).HandlerType

proc `$`*(x: posix.SocketHandle): string {.inline.} = $(x.cint)
proc `$`*(x: HandlerType): string {.inline.} = $(x.int)
proc `==`*(x, y: HandlerType): bool {.borrow.}


var serveridlock: Lock
initLock(serveridlock)
var nextserverid: int
proc initGuildenServer*(gs: var GuildenServer, handlers: openarray[HandlerAssociation]) {.gcsafe, nimcall.} =
  gs.portcount = handlers.len
  for i in 0 ..< gs.portcount:
    gs.porthandlers[i] = handlers[i]
  withLock(serveridlock):
    gs.serverid = nextserverid
    nextserverid += 1


proc signalSIGINT*() =
  discard kill(getCurrentProcessId().cint, SIGINT)


#proc registerDefaultHandler*(gs: GuildenServer,  handlertype: HandlerType, callback: HandlerCallback) =
#  gs.handlercallbacks[handlertype] = callback



proc registerHandler*(gs: GuildenServer,  handlertype: HandlerType, callback: HandlerCallback) =
  assert(handlertype.int > 0, "ctx types below 1 are reserved for internal use")
  gs.handlercallbacks[handlertype] = callback


#proc registerTimerhandler*(gs: GuildenServer, callback: proc() {.gcsafe, nimcall, raises: [].}, interval: int) =
#  gs.timerHandler = callback
#  discard gs.selector.registerTimer(interval, false, SocketData(fdKind: Ticker, dataobject: nil))



proc registerErrorhandler*(gs: GuildenServer, callback: ErrorCallback) =
  gs.errorHandler = callback


proc registerShutdownhandler*(gs: GuildenServer, callback: proc() {.gcsafe, nimcall, raises: [].}) =
  gs.shutdownHandler = callback


proc handleRead*(gs: ptr GuildenServer, data: ptr SocketData) =
  assert(gs.handlercallbacks[data.handlertype] != nil, "No ctx registered for HandlerType " & $data.handlertype)
  gs.handlercallbacks[data.handlertype](gs, data)


proc closeFd*(gs: ptr GuildenServer, fd: posix.SocketHandle) {.raises: [].} =
  try:
    if gs.selector.contains(fd): gs.selector.unregister(fd)
    fd.close()
  except: discard