from selectors import Selector, contains, registerTimer, unregister
from nativesockets import SocketHandle, close
from net import Port
from os import getCurrentProcessId
from posix import kill, SIGINT
from streams import StringStream, getPosition
from locks import Lock


const
  NullHandle* = (-2147483647)

  MaxHttpHeaderFields* {.intdefine.} = 25
  MaxHttpHeaderValueLength* {.intdefine.} = 200
  MaxRequestLength* {.intdefine.} = 1000
  MaxResponseLength* {.intdefine.} = 100000
  RcvTimeOut* {.intdefine.} = 5 # SO_RCVTIMEO, https://linux.die.net/man/7/socket

type
  Clientid* = distinct int32

  ServerState* = enum
    Initializing, Normal, Maintenance, Shuttingdown

  FdKind* = enum
      Signal, Ticker, Server, Http, Ws

  Data* = object
    fdKind*: FdKind
    clientid*: Clientid

  GuildenServer* {.inheritable.} = ref object
    tcpport*: Port
    serverstate*: ServerState
    turbo*: bool
    threadcount*: int
    headerfieldarray* : array[MaxHttpHeaderFields, string]
    lastheader*: int
    selector*: Selector[Data]
    inflight*: int
    httpHandler*: proc(c: GuildenVars) {.gcsafe, raises: [].}
    wsHandler*: proc(c: GuildenVars) {.gcsafe, raises: [].}
    wsdisconnectHandler*: proc(c: GuildenVars, closed: bool) {.gcsafe, raises: [].}
    timerHandler*: proc() {.gcsafe, raises: [].}
    errorHandler*: proc(c: GuildenVars, msg: string) {.gcsafe, raises: [].}
    shutdownHandler*: proc() {.gcsafe, raises: [].}
    ctxlock*: Lock

  
  GuildenVars* {.inheritable.} = ref object
    gs*: GuildenServer
    threadid*: int
    fd*: SocketHandle
    clientid*: Clientid
    path*: int
    pathlen*: int
    methlen*: int
    recvbuffer* : StringStream
    wsrecvheader* : array[8, char]
    headervalues* : array[MaxHttpHeaderFields, string]
    wsheader* : StringStream  
    bodystartpos* : int   
    sendbuffer* : StringStream
    currentexceptionmsg* : string


proc initContext*(c: GuildenVars) =  
  c.threadid = -1

    
proc `$`*(x: posix.SocketHandle): string {.inline.} = $(x.cint)

proc `==`*(x, y: Clientid): bool {.borrow.}

proc `$`*(x: Clientid): string {.inline.} = $(x.int32)


proc signalSIGINT*() =
  discard kill(getCurrentProcessId().cint, SIGINT)


proc registerHttphandler*(gs: GuildenServer, callback: proc(c: GuildenVars) {.gcsafe, nimcall, raises: [].}, headerfields: openarray[string] = ["content-length"]) =
  gs.httpHandler = callback
  for i in 0 .. headerfields.high: gs.headerfieldarray[i] = headerfields[i]
  gs.lastheader = headerfields.high


proc registerWshandler*(gs: GuildenServer, callback: proc(c: GuildenVars) {.gcsafe, nimcall, raises: [].}) =
  gs.wsHandler = callback


proc registerWsdisconnecthandler*(gs: GuildenServer, callback: proc(c: GuildenVars, closedbyclient: bool) {.gcsafe, nimcall, raises: [].}) =
  gs.wsdisconnectHandler = callback


proc registerTimerhandler*(gs: GuildenServer, callback: proc() {.gcsafe, nimcall, raises: [].}, intervalsecs: int) =
  gs.timerHandler = callback
  discard gs.selector.registerTimer(1000 * intervalsecs, false, Data(fdKind: Ticker, clientid: NullHandle.Clientid))


proc registerErrorhandler*(gs: GuildenServer, callback: proc(c: GuildenVars, msg: string) {.gcsafe, nimcall, raises: [].}) =
  gs.errorHandler = callback
  

proc registerShutdownhandler*(gs: GuildenServer, callback: proc() {.gcsafe, nimcall, raises: [].}) =
  gs.shutdownHandler = callback


proc upgradeHttpToWs*(c: GuildenVars, clientid: Clientid) =
  c.clientid = clientid


proc closeFd*(gs: GuildenServer, fd: posix.SocketHandle) =
  if fd.int == NullHandle: return
  try:
    if gs.selector.contains(fd): gs.selector.unregister(fd)
    fd.close()
  except: discard


proc getPath*(c: GuildenVars): string =
  if c.pathlen == 0: return
  return c.recvbuffer.data[c.path ..< c.path + c.pathlen]

proc isPath*(c: GuildenVars, apath: string): bool =
  if c.pathlen != apath.len: return false
  for i in 0 ..< c.pathlen:
    if c.recvbuffer.data[c.path + i] != apath[i]: return false
  return true

proc pathStarts*(c: GuildenVars, pathstart: string): bool =
  if c.pathlen < pathstart.len: return false
  for i in 0 ..< pathstart.len:
    if c.recvbuffer.data[c.path + i] != pathstart[i]: return false
  return true

proc getMethod*(c: GuildenVars): string =
  if c.methlen == 0: return
  return c.recvbuffer.data[0 ..< c.methlen]

proc isMethod*(c: GuildenVars, amethod: string): bool =
  if c.methlen != amethod.len: return false
  for i in 0 ..< c.methlen:
    if c.recvbuffer.data[i] != amethod[i]: return false
  return true

proc getHeader*(c: GuildenVars, header: string): string {.inline.} =
  let tokeni = c.gs.headerfieldarray.find(header)
  if tokeni == -1: return
  return c.headervalues[tokeni]
  
proc getBody*(c: GuildenVars): string =
  try: result = c.recvbuffer.data[c.bodystartpos ..< c.recvbuffer.getPosition()]
  except: doaSsert(false, "getBody error: " & getCurrentExceptionMsg()) 