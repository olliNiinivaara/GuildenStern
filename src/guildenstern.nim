#
#
#   Guildenstern
#
##  multithreading plug'n'play posix http server
#
#   (c) Copyright 2020 Olli Niinivaara
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#  
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
#

from posix import SocketHandle
export SocketHandle

when not defined(nimdoc):
  import guildenstern/guildenserver
  import guildenstern/dispatcher
  # import guildenstern/httpout
  # import guildenstern/wsout

  export GuildenServer, ServerState, serve, signalSIGINT
  export Handler
  export newGuildenServer, registerHandler, registerShutdownhandler
  #export write
  # export sendToWs
else:
  from net import Port
  from httpcore import HttpCode, Http200
  from streams import StringStream


  type
    GuildenServer* {.inheritable.} = ref object
      ## | Contains global variables.
      ## | This is available in GuildenVars via ``gs`` property.
      ## | Inherit and add custom properties as needed.
      tcpport*: Port ## set when calling `serve`

    GuildenVars* {.inheritable.} = ref object
      ## THIS DOC IS OUTDATED!!
      ## | Contains thread-local variables, acting as the context for an incoming request.
      ## | This is available in callbacks as parameter.
      ## | Inherit and add custom properties as needed.
      gs*: GuildenServer
      clientid*: Clientid
      fd*: SocketHandle ## fd = "file descriptor" (trad., of Unix folklore origin)
      recvbuffer* : StringStream ## Contains the request
      bodystartpos* : int ## Start index of body part in HTTP requests
      sendbuffer* : StringStream ## Response data to send with `reply` or `sendToWs`
  
    Clientid* = distinct int32
      ## | Value to identify WebSocket client of the request.
      ## | Set this in upgradeHttpToWs proc and it will thereafter be available as ``clientid`` in GuildenVars.
    
  const
    NullHandle = (-2147483647)

    MaxHttpHeaderFields* {.intdefine.} = 25
      ## Compile time define pragma to set maximum number of captured http headers.
    MaxHttpHeaderValueLength* {.intdefine.} = 200
      ## Compile time define pragma to set maximum length for any http header value.
    MaxRequestLength* {.intdefine.} = 1000
      ## Compile time define pragma to limit size of receivable requests (maximum size of ``recvbuffer``).
    MaxResponseLength* {.intdefine.} = 100000
      ## Compile time define pragma to limit size of sendable responses (maximum size of ``sendbuffer``).
    RcvTimeOut* {.intdefine.} = 5
      ## Compile time define pragma to  set sockets timeout SO_RCVTIMEO, https://linux.die.net/man/7/socket


  proc serve*[T: GuildenVars](gs: GuildenServer, port: int) =
    ## Starts the server main loop which runs until SIGINT is received.
    ##
    ## Give type of GuildenVars so that you can use your own customized request context.
    ## (Note: method call syntax does not work, see `#14254 <https://github.com/nim-lang/Nim/issues/14254>`_)
    ## 
    ## Give tcp port as parameter.
    ## 
    ## **Example:**
    ##
    ## .. code-block:: Nim
    ##
    ##  import guildenstern
    ##   
    ##  type
    ##    MyGuildenServer = ref object of GuildenServer
    ##      customserverproperty: string
    ##
    ##    MyGuildenVars = ref object of GuildenVars
    ##      customcontextproperty: string
    ##   
    ##  proc onRequest(h: MyGuildenVars) =
    ##    h.customcontextproperty = "world"
    ##    h.reply(((MyGuildenServer)h.gs).customserverproperty & h.customcontextproperty)
    ##  
    ##  let server = new MyGuildenServer
    ##  server.customserverproperty = "hello,"
    ##  server.registerHttphandler(onRequest, [])
    ##  serve[MyGuildenVars](server, 8080)

  proc signalSIGINT*() =
    ## Sends Ctrl-C to current process which is caught by GuildenServer and graceful shutdown is commenced.
    discard

  proc registerHttphandler*(gs: GuildenServer, callback: proc(h: GuildenVars) {.gcsafe, nimcall, raises: [].},
   headerfields: openarray[string] = ["content-length"]) =
    ## Registers the callback procedure to handle incoming http requests. Give list of header field names to capture.
    ## If body is to be received, "content-length" must be one of the capturable fields.
    ## Receiving duplicate header fields is not supported.
    discard

  proc registerWshandler*(gs: GuildenServer, callback: proc(h: GuildenVars) {.gcsafe, nimcall, raises: [].}) =
    ## Registers the callback procedure to handle incoming WebSocket requests.
    discard

  proc registerWsdisconnecthandler*(gs: GuildenServer, callback: proc(h: GuildenVars, closedbyclient: bool) {.gcsafe, nimcall, raises: [].}) =
    ## Registers the callback procedure to handle disconnecting WebSocket requests.
    ## If closedbyclient == true, client closed the session and usually must not be allowed to login again without credentials.
    ## If closedbyclient == false, socket was lost due to a network problem and usually the client will try to reconnect soon.
    discard

  proc registerTimerhandler*(gs: GuildenServer, callback: proc() {.gcsafe, nimcall, raises: [].}, intervalsecs: int) =
    ## Registers the callback procedure to handle timer events with given interval in seconds.
    discard

  proc registerErrorhandler*(gs: GuildenServer, callback: proc(h: GuildenVars, msg: string) {.gcsafe, nimcall, raises: [].}) =
    ## Registers the callback procedure to handle error conditions.
    ## If serverstate is set to Maintenance, all http and ws requests will call this proc instead of their normal handler.
    ## If proc is called due to a genuine error, error message is available in msg.
    discard
    

  proc registerShutdownhandler*(gs: GuildenServer, callback: proc() {.gcsafe, nimcall, raises: [].}) =
    ## Registers the callback procedure to perform application shutdown like closing database connections.
    discard


  proc upgradeHttpToWs*(h: GuildenVars, clientid: Clientid) =
    ## Upgrades the current file descriptor to WebSocket protocol.
    ## Give a clientid that allows identification of the connected user.
    ## Note that GuildenStern does not offer a way to get ``fd`` s of connected clients, you have to store these mappings yourself.
    discard


  proc getPath*(h: GuildenVars): string =
    ## Returns deep copy of http path.
    discard
    
  proc isPath*(h: GuildenVars, apath: string): bool =
    ## Checks if http path is apath.
    discard

  proc pathStarts*(h: GuildenVars, pathstart: string): bool =
    ## Checks if http path starts with pathstart.
    discard

  proc getMethod*(h: GuildenVars): string =
    ## Returns deep copy of http method.
    discard

  proc isMethod*(h: GuildenVars, amethod: string): bool =
    ## Checks if http method is amethod.
    discard

  proc getHeader*(h: GuildenVars, header: string): string {.inline.} =
    ## Returns value of header.
    discard

  proc getHeaders*(h: GuildenVars): string {.inline.} =
    ## Returns deep copy of headers part in http request.
    discard

  proc getBody*(h: GuildenVars): string =
    ## Returns deep copy of http body / web socket request.
    discard

  proc isBody*(h: GuildenVars, abody: string): bool =
    ## Checks if http body / web socket request is abody.
    discard

  proc write*(h: GuildenVars, str: string): bool {.raises: [].} =
    ## appends string to sendbuffer, returns false if append failed.
    discard

  proc reply*(h: GuildenVars, code: HttpCode, body: string, headers="") {.inline.} =
    discard

  proc reply*(h: GuildenVars, code: HttpCode, body: string, headers: openArray[string]) {.inline.} =
    discard

  proc reply*(h: GuildenVars, code: HttpCode, body: string, headers: seq[string]) {.inline.} =
    discard

  proc reply*(h: GuildenVars, code: HttpCode, body: string,  headers: openArray[seq[string]]) {.inline.} =
    discard

  proc reply*(h: GuildenVars, body: string, code=Http200) {.inline.} =
    discard

  proc replyHeaders*(h: GuildenVars, headers: openArray[string], code: HttpCode=Http200) {.inline.} =
    discard

  proc replyHeaders*(h: GuildenVars, headers: seq[string], code: HttpCode=Http200) {.inline.} =
    discard

  proc replyHeaders*(h: GuildenVars, headers: openArray[seq[string]], code: HttpCode=Http200) {.inline.} =
    discard

  proc reply*(h: GuildenVars, headers: openArray[string]) {.inline.} =
    ## Responds with contents of sendbuffer
    discard

  proc reply*(h: GuildenVars, headers: seq[string]) {.inline.} =
    ## Responds with contents of sendbuffer
    discard 

  proc reply*(h: GuildenVars, headers: openArray[seq[string]]) {.inline.} =
    ## Responds with contents of sendbuffer
    discard

  proc replyCode*(h: GuildenVars, code: HttpCode) {.inline.} =
    discard
  

  proc sendToWs*(h: GuildenVars, toSocket = NullHandle.SocketHandle, text: StringStream = nil): bool =
    ## Sends text data to a websocket.
    ## 
    ## If text parameter is not given, sends contents of ``sendbuffer``.
    ## 
    ## If toSocket parameter is not given, replies back to socket that made the request.
    ## 
    ## Returns false if sending failed.
    ## 
    ## Do not write to same socket from many threads in parallel, use some serialization strategy like locking.
    discard