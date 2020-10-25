from selectors import Selector, contains, registerTimer, unregister
from nativesockets import SocketHandle, close
from net import Port
from os import getCurrentProcessId
from posix import kill, SIGINT
import locks


const
  NullHandle* = (-2147483647)
  RcvTimeOut* {.intdefine.} = 5 # SO_RCVTIMEO, https://linux.die.net/man/7/socket
  
when compileOption("threads"):
  import threadpool
  const MaxHandlerCount* {.intdefine.} = MaxThreadPoolSize
else:
  const MaxHandlerCount* = 1

type
  HandlerType* = distinct int
  
  ServerState* = enum
    Initializing, Normal, Maintenance, Shuttingdown

  SocketData* = object
    socket*: SocketHandle
    handlertype*: HandlerType
    dataobject*: ref RootObj

  GuildenServer* {.inheritable.} = ref object
    serverid*: int
    tcpport*: Port
    multithreading*: bool
    serverstate*: ServerState
    selector*: Selector[SocketData]
    defaulthandlertype*: HandlerType    
    handlercallbacks*: array[0.HandlerType .. 200.HandlerType, HandlerCallback]
    shutdownHandler*: proc() {.gcsafe, raises: [].}

  Handler* {.inheritable.} = ref object of RootObj
    handlertype*: HandlerType
    gs*: GuildenServer
    socket*: SocketHandle
    dataobject*: ref RootObj
    ishandling*: bool
    currentexceptionmsg* : string
  
  HandlerCallback* = proc(gs: ptr GuildenServer, data: ptr SocketData){.gcsafe, nimcall, raises: [].}

const
  InvalidHandlerType* = 0.HandlerType
  HttpHandlerType* = 101.HandlerType
  CustomHandlerType* = 102.HandlerType  
  ServerHandlerType* = 103.HandlerType
  TimerHandlerType* = 104.HandlerType
  SignalHandlerType* = 105.HandlerType
  

proc `$`*(x: posix.SocketHandle): string {.inline.} = $(x.cint)
proc `$`*(x: HandlerType): string {.inline.} = $(x.int)
proc `==`*(x, y: HandlerType): bool {.borrow.}


var serveridlock: Lock
initLock(serveridlock)
var nextserverid: int
proc newGuildenServer*(defaulthandlertype: HandlerType = HttpHandlerType): GuildenServer {.gcsafe, nimcall.} =
  result = new GuildenServer
  result.defaulthandlertype = defaulthandlertype
  withLock(serveridlock):
    result.serverid = nextserverid
    nextserverid += 1


proc signalSIGINT*() =
  discard kill(getCurrentProcessId().cint, SIGINT)


proc registerDefaultHandler*(gs: GuildenServer,  handlertype: HandlerType, callback: HandlerCallback) =
  gs.handlercallbacks[handlertype] = callback


proc registerHandler*(gs: GuildenServer,  handlertype: HandlerType, callback: HandlerCallback) =
  assert(handlertype != InvalidHandlerType, "handler type 0 is not to be used")
  assert(handlertype.int < 100, "handler types over 100 are reserved for internal use")
  gs.handlercallbacks[handlertype] = callback


#proc registerTimerhandler*(gs: GuildenServer, callback: proc() {.gcsafe, nimcall, raises: [].}, interval: int) =
#  gs.timerHandler = callback
#  discard gs.selector.registerTimer(interval, false, SocketData(fdKind: Ticker, dataobject: nil))


proc registerShutdownhandler*(gs: GuildenServer, callback: proc() {.gcsafe, nimcall, raises: [].}) =
  gs.shutdownHandler = callback


proc handleRead*(gs: ptr GuildenServer, data: ptr SocketData) =
  assert(gs.handlercallbacks[data.handlertype] != nil, "No handler registered for HandlerType " & $data.handlertype)
  gs.handlercallbacks[data.handlertype](gs, data)


proc closeFd*(gs: GuildenServer, fd: posix.SocketHandle) {.raises: [].} =
  if fd.int == NullHandle: return
  try:
    if gs.selector.contains(fd): gs.selector.unregister(fd)
    fd.close()
  except: discard