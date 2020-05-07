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
    httpHandler*: proc(gv: GuildenVars) {.gcsafe, raises: [].}
    wsHandler*: proc(gv: GuildenVars) {.gcsafe, raises: [].}
    wsdisconnectHandler*: proc(gv: GuildenVars, closed: bool) {.gcsafe, raises: [].}
    timerHandler*: proc() {.gcsafe, raises: [].}
    errorHandler*: proc(gv: GuildenVars, msg: string) {.gcsafe, raises: [].}
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


proc initGuildenVars*(gv: GuildenVars) =  
  gv.threadid = -1

    
proc `$`*(x: posix.SocketHandle): string {.inline.} = $(x.cint)

proc `==`*(x, y: Clientid): bool {.borrow.}

proc `$`*(x: Clientid): string {.inline.} = $(x.int32)
 

proc signalSIGINT*() =
  discard kill(getCurrentProcessId().cint, SIGINT)


proc registerHttphandler*(gs: GuildenServer, callback: proc(gv: ref object){.gcsafe, nimcall, raises: [].}, headerfields: openarray[string] = ["content-length"]) =
  gs.httpHandler = cast[proc(gv: GuildenVars){.gcsafe, nimcall, raises: [].}](callback)
  for i in 0 .. headerfields.high: gs.headerfieldarray[i] = headerfields[i]
  gs.lastheader = headerfields.high


proc registerWshandler*(gs: GuildenServer, callback: proc(gv: ref object) {.gcsafe, nimcall, raises: [].}) =
  gs.wsHandler = cast[proc(gv: GuildenVars){.gcsafe, nimcall, raises: [].}](callback)


proc registerWsdisconnecthandler*(gs: GuildenServer, callback: proc(gv: ref object, closedbyclient: bool) {.gcsafe, nimcall, raises: [].}) =
  gs.wsdisconnectHandler = cast[proc(gv: GuildenVars){.gcsafe, nimcall, raises: [].}](callback)


proc registerTimerhandler*(gs: GuildenServer, callback: proc() {.gcsafe, nimcall, raises: [].}, intervalsecs: int) =
  gs.timerHandler = callback
  discard gs.selector.registerTimer(1000 * intervalsecs, false, Data(fdKind: Ticker, clientid: NullHandle.Clientid))


proc registerErrorhandler*(gs: GuildenServer, callback: proc(gv: ref object, msg: string) {.gcsafe, nimcall, raises: [].}) =
  gs.errorHandler = cast[proc(gv: GuildenVars, msg: string){.gcsafe, nimcall, raises: [].}](callback)
  

proc registerShutdownhandler*(gs: GuildenServer, callback: proc() {.gcsafe, nimcall, raises: [].}) =
  gs.shutdownHandler = callback


proc upgradeHttpToWs*(gv: ref object, clientid: Clientid) =
  (GuildenVars)(gv).clientid = clientid


proc closeFd*(gs: GuildenServer, fd: posix.SocketHandle) =
  if fd.int == NullHandle: return
  try:
    if gs.selector.contains(fd): gs.selector.unregister(fd)
    fd.close()
  except: discard


#[proc registerWshandler*(gs: GuildenServer, callback: proc(gv: GuildenVars) {.gcsafe, nimcall, raises: [].}) =
  gs.wsHandler = callback


proc registerWsdisconnecthandler*(gs: GuildenServer, callback: proc(gv: GuildenVars, closedbyclient: bool) {.gcsafe, nimcall, raises: [].}) =
  gs.wsdisconnectHandler = callback


proc registerTimerhandler*(gs: GuildenServer, callback: proc() {.gcsafe, nimcall, raises: [].}, intervalsecs: int) =
  gs.timerHandler = callback
  discard gs.selector.registerTimer(1000 * intervalsecs, false, Data(fdKind: Ticker, clientid: NullHandle.Clientid))


proc registerErrorhandler*(gs: GuildenServer, callback: proc(gv: GuildenVars, msg: string) {.gcsafe, nimcall, raises: [].}) =
  gs.errorHandler = callback
  

proc registerShutdownhandler*(gs: GuildenServer, callback: proc() {.gcsafe, nimcall, raises: [].}) =
  gs.shutdownHandler = callback


proc upgradeHttpToWs*(gv: GuildenVars, clientid: Clientid) =
  gv.clientid = clientid


proc closeFd*(gs: GuildenServer, fd: posix.SocketHandle) =
  if fd.int == NullHandle: return
  try:
    if gs.selector.contains(fd): gs.selector.unregister(fd)
    fd.close()
  except: discard]#


proc getPath*(gv: GuildenVars): string =
  if gv.pathlen == 0: return
  return gv.recvbuffer.data[gv.path ..< gv.path + gv.pathlen]

proc isPath*(gv: GuildenVars, apath: string): bool =
  if gv.pathlen != apath.len: return false
  for i in 0 ..< gv.pathlen:
    if gv.recvbuffer.data[gv.path + i] != apath[i]: return false
  return true

proc pathStarts*(gv: GuildenVars, pathstart: string): bool =
  if gv.pathlen < pathstart.len: return false
  for i in 0 ..< pathstart.len:
    if gv.recvbuffer.data[gv.path + i] != pathstart[i]: return false
  return true

proc getMethod*(gv: GuildenVars): string =
  if gv.methlen == 0: return
  return gv.recvbuffer.data[0 ..< gv.methlen]

proc isMethod*(gv: GuildenVars, amethod: string): bool =
  if gv.methlen != amethod.len: return false
  for i in 0 ..< gv.methlen:
    if gv.recvbuffer.data[i] != amethod[i]: return false
  return true

proc getHeader*(gv: GuildenVars, header: string): string {.inline.} =
  let tokeni = gv.gs.headerfieldarray.find(header)
  if tokeni == -1: return
  return gv.headervalues[tokeni]
  
proc getBody*(gv: GuildenVars): string =
  try: result = gv.recvbuffer.data[gv.bodystartpos ..< gv.recvbuffer.getPosition()]
  except: doaSsert(false, "getBody error: " & getCurrentExceptionMsg()) 