const GuildenSternVersion* = "5.0.0"

#   Guildenstern
#
#  Modular multithreading Linux HTTP + WebSocket server
#
#  (c) Copyright 2020-2021 Olli Niinivaara
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#  
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
## A modular multithreading Linux HTTP + WebSocket server.
## 
## Example
## =======
## 
## In this example port number is coded into html just for demonstration purposes. In reality, use your
## reverse proxy to route different types of requests to different ports.
## 
## .. code-block:: Nim
##
##    # nim c -r --gc:arc --d:release --threads:on --d:threadsafe example.nim
##    
##    import cgi, guildenstern/[ctxheader, ctxfull]
##
##    proc handleGet(ctx: HttpCtx) =
##      let html = """
##        <!doctype html><title>GuildenStern Example</title><body>
##        <form action="http://localhost:5051" method="post">
##        <input name="say" id="say" value="Hi"><button>Send"""
##      ctx.reply(Http200, html)
##        
##    proc handlePost(ctx: HttpCtx, headers: StringTableRef) =
##      try:
##        echo readData(ctx.getBody()).getOrDefault("say")
##        ctx.reply(Http303, ["location: " & headers.getOrDefault("origin")])
##      except: ctx.reply(Http500)
##           
##    var server = new GuildenServer
##    server.initHeaderCtx(handleGet, 5050, false)
##    server.initFullCtx(handlePost, 5051)
##    echo "GuildenStern HTTP server serving at localhost:5050"
##    server.serve()
## 
## See also
## ========
## 
## | `ctxtimer <http://olliNiinivaara.github.io/GuildenStern/ctxtimer.html>`_
## | `ctxheader <http://olliNiinivaara.github.io/GuildenStern/ctxheader.html>`_
## | `ctxbody <http://olliNiinivaara.github.io/GuildenStern/ctxbody.html>`_
## | `ctxfull <http://olliNiinivaara.github.io/GuildenStern/ctxfull.html>`_
## | `ctxstream <http://olliNiinivaara.github.io/GuildenStern/ctxstream.html>`_
## | `ctxws <http://olliNiinivaara.github.io/GuildenStern/ctxws.html>`_
##


when not defined(nimdoc):
  import guildenstern/guildenserver
  export guildenserver
  import guildenstern/dispatcher
  export serve, registerThreadInitializer, getLoads
  import guildenstern/ctxhttp
  export ctxhttp
else:
  import strtabs, httpcore
  from posix import SocketHandle
  const    
    MaxHeaderLength* {.intdefine.} = 10000
      ## Maximum acceptable character length for HTTP header.
      ## 
      ## This is modifiable with a `compile-time define <https://nim-lang.org/docs/manual.html#implementation-specific-pragmas-compileminustime-define-pragmas>`_.
    
    MaxRequestLength* {.intdefine.} = 100000
      ## Maximum acceptable length for a received message.
      ## | For normal HTTP handlers, this denotes header length + content length
      ## | For streaming handlers, this denotes chunk size
      ## | For websockets this denotes payload size
      ## 
      ## If a client tries to send more than this, socket will be immediately closed with `ProtocolViolated` close cause
      ## 
      ## This is modifiable with a `compile-time define <https://nim-lang.org/docs/manual.html#implementation-specific-pragmas-compileminustime-define-pragmas>`_.
  
 
  type
    LogLevel* = enum TRACE, DEBUG, INFO, NOTICE, WARN, ERROR, FATAL, NONE

    CtxId = distinct int

    GuildenServer* {.inheritable.} = ref object
      ## | This is available in request context as ``gs`` property.
      ## | You can inherit this and add custom properties as needed.
   
    SocketData* = object
      ## | Data associated with a request context.
      ## | This is available in request context as ``socketdata`` property.
      port*: uint16
      socket*: posix.SocketHandle
      ctxid*: CtxId
      customdata*: pointer

    Ctx* {.inheritable, shallow.} = ref object
      ## | Common abstract base class for all request contexts.
      ## | A request context is received as a parameter in request callbacks. 
      gs*: ptr GuildenServer
      socketdata*: ptr SocketData

    HttpCtx* = ref object of Ctx
      ## | Common abstract base class for all HTTP handling request contexts.

    SocketCloseCause* = enum
      ## Parameter in CloseCallback.
      CloseCalled ## available for applications to use when calling closeSocket
      SecurityThreatened ## available for applications to use when calling closeSocket      
      ClosedbyClient ## Client closed the connection
      ConnectionLost ## TCP/IP connection was dropped
      TimedOut ## Client did not send/receive all expected data
      ProtocolViolated ## Client was sending garbage
      AlreadyClosed ## Another thread has closed the socket
      NetErrored ## Some operating system level error happened
      Excepted ## A Nim exception happened
    
    CloseCallback* = proc(ctx: Ctx, socket: SocketHandle, cause: SocketCloseCause, msg: string){.gcsafe, nimcall, raises: [].}
      ## Called whenever a socket is closed. The closed socket is always given as `socket` parameter. Moreover:
      ##  
      ## | if ctx.socketdata.socket == socket: request processing thread is running and closed socket is the requester.
      ## | if ctx.socketdata.socket != socket: request processing thread is running but closed socket is some other socket. This is most usually the case when a connection is lost to websocket that the requester was sending messages to.
      ## | ctx.socketdata.socket == INVALID_SOCKET: dispatcher noticed that socket was closed and spawned a ctx just for calling this callback. In this case Ctx is of generic type Ctx, but `getProtocolName` lets you check the type of the closed socket.


  proc registerThreadInitializer*(callback: proc() {.nimcall, gcsafe, raises: [].}) =
    ## Registers a procedure in which you can initialize threadvars. This is called exactly once for each thread when first server starts serving.
    ##
    ## **Example:**
    ##
    ## .. code-block:: Nim
    ##
    ##    import guildenstern/ctxtimer, os, random
    ##
    ##    var threadcount: int
    ##    var rounds: int
    ##    var x {.threadvar.}: int
    ##
    ##    proc initializeThreadvars() =
    ##      x = threadcount.atomicInc
    ##      echo "new thread initialized: ", x
    ##
    ##    proc tiktok() =
    ##      echo "tik ", x
    ##      sleep(500 + rand(2000))
    ##      echo "tok ", x
    ##      if rounds.atomicInc > 50: shutdown()
    ##
    ##    randomize()
    ##    registerThreadInitializer(initializeThreadvars)
    ##    var server = new GuildenServer
    ##    server.initTimerCtx(100, tiktok)
    ##    server.serve()
    ##    echo "----"
    ##    echo "waiting for processes to finish..."
    ##    sleep(2510)
    ##    echo "graceful shutdown handling here!"
    discard

  proc serve*(gs: GuildenServer, threadcount = -1, loglevel = NOTICE) {.gcsafe.} =
    ## Starts server event dispatcher loop which runs until shutdown() is called or SIGINT received.
    ## | Worker thread count can be set; 1 means single-threaded mode and -1 (default) will use processor core count + 2.
    ## | Possible log levels are: TRACE, DEBUG, INFO, NOTICE, WARN, ERROR, FATAL, NONE.
    ## 
    ## **Example:**
    ##
    ## .. code-block:: Nim
    ##
    ##    import guildenstern/ctxheader  
    ##    var server = new GuildenServer
    ##    server.initHeaderCtx(proc(ctx: HttpCtx) = ctx.reply(Http200), 5050)
    ##    server.serve(1, TRACE)
    discard

  proc setLogger*(gs: var GuildenServer, logger: proc(loglevel: LogLevel, message: string, exception: ref Exception) {.gcsafe, nimcall, raises: [].}) =
    ## Change the default logging procedure.
    discard


  proc registerConnectionclosedhandler*(gs: GuildenServer, callback: CloseCallback) =
      ## Registers procedure that is called whenever socket connection is closed.
      discard

  proc closeSocket*(ctx: Ctx, cause = CloseCalled, msg = "") {.raises: [].} =
      ## Closes the socket of the current requester.
      ## | `msg` can be any further info that will be delivered to registered CloseCallback.
      discard
  
  proc closeOtherSocket*(gs: GuildenServer, socket: SocketHandle, cause: SocketCloseCause, msg: string = "") {.raises: [].} =
    ## Closes arbitrary socket.
    discard

  proc getProtocolName*(ctx: Ctx): string =
    ## Helper proc to use in closecallbacks to get type of the closed socket when closed socket was not in request context.
    ## For websockets returns "websocket" and for other default handlers "http". Your custom handlers can register other protocol names.
    discard

  proc getLoads*(): (int, int, int) =
    ## returns load statistics over all guildenservers as tuple, where:
    ## | 0 = current load; amount of currently active threads, i.e. requests being currently simultaneously served
    ## | 1 = peak load; maximum current load since current load was zero
    ## | 2 = max load; maximum load since first server was started    
    discard
  
  proc shutdown*() =
    ## Cancels pending network I/O and breaks event dispatcher loops of all guildenservers. Sending SIGINT / pressing Ctrl-C will automatically call shutdown().
    ## | See `registerThreadInitializer` for example.
    discard

  proc getUri*(ctx: HttpCtx): string {.raises: [].} =
    ## Request URI of a HttpCtx.
    discard

  proc isUri*(ctx: HttpCtx, uri: string): bool {.raises: [].} =
    ## true if request URI is uri.
    discard

  proc startsUri*(ctx: HttpCtx, uristart: string): bool {.raises: [].} =
    ## true if request URI starts with uristart.
    discard

  proc getMethod*(ctx: HttpCtx): string {.raises: [].} =
    ## Request method of a HttpCtx.
    discard

  proc isMethod*(ctx: HttpCtx, amethod: string): bool {.raises: [].} =
    ## true if request method is amethod.
    discard

  proc getHeaders*(ctx: HttpCtx): string =
    ## Returns the header part of request as string.
    discard

  proc getBodystart*(ctx: HttpCtx): int {.inline.} =
    ## Returns the start position of body in ``request``. If there's no body, returns -1.
    discard

  proc getBodylen*(ctx: HttpCtx): int =
    ## Returns the length of request body.
    discard

  proc getBody*(ctx: HttpCtx): string =
    ## Returns the body part of request as string.
    discard

  proc isBody*(ctx: HttpCtx, body: string): bool =
    ## true if request body is body.
    discard

  proc getRequest*(ctx: HttpCtx): string =
    ## Returns the whole request as string.

  proc isRequest*(ctx: HttpCtx, request: string): bool =
    ## true if request body is body.
    discard

  proc parseHeaders*(ctx: HttpCtx, fields: openArray[string], toarray: var openArray[string]) =
    ## Parses header `fields` values into `toarray`.
    ## 
    ## **Example:**
    ##
    ## .. code-block:: Nim
    ##
    ##    import guildenstern/ctxheader
    ##    import httpclient
    ##       
    ##    const headerfields = ["afield", "anotherfield"]
    ##    var headers {.threadvar.}: array[2, string]
    ##    var client {.threadvar.}: HttpClient
    ##     
    ##    proc initializeThreadvars() =
    ##      try:
    ##        client = newHttpClient()
    ##        client.headers = newHttpHeaders(
    ##          { "afield": "afieldvalue", "bfield": "bfieldvalue" })
    ##      except: echo getCurrentExceptionMsg()
    ##     
    ##    proc sendRequest() =
    ##      try: discard client.request("http://localhost:5050")
    ##      except: echo getCurrentExceptionMsg()
    ##     
    ##    proc onRequest(ctx: HttpCtx) =
    ##      ctx.parseHeaders(headerfields, headers)
    ##      echo headers
    ##      ctx.reply(Http204)
    ##     
    ##    doAssert(compileOption("threads"), """this example must be compiled with threads;
    ##      "nim c -r --gc:arc --threads:on --d:threadsafe example.nim"""")
    ##    registerThreadInitializer(initializeThreadvars)
    ##    var server = new GuildenServer
    ##    server.initHeaderCtx(onRequest, 5050)
    ##    server.registerTimerhandler(sendRequest, 1000)
    ##    server.serve()
    discard
    
  proc parseHeaders*(ctx: HttpCtx, headers: StringTableRef) =
    ## Parses all headers into given strtabs.StringTable.
    discard

  template reply*(ctx: HttpCtx, code: HttpCode, body: string, headers: openArray[string]) =
    ## Sends a reply to the socket in context. Aborts silently if socket is closed (there's not much to do).
    ## Note that body has to be addressable for posix, use lets instead of consts.
    ## Syntactic sugar varations of this template below.
    ## If you need even more control over replying, see procs in httpresponse module.
    discard

  template reply*(ctx: HttpCtx, code: HttpCode, headers: openArray[string]) =    
    discard

  template reply*(ctx: HttpCtx, body: string) =
    ## Http200 Ok
    discard

  template reply*(ctx: HttpCtx,  code: HttpCode, body: string) =
    discard

  template replyStart*(ctx: HttpCtx, code: HttpCode, contentlength: int, firstpart: string, headers: openArray[string]): bool =
    ## Indicates that reply will be given in multiple chunks. This first part submits the HTTP status code, content-length,
    ## first part of body and headers.
    ## Zero or more replyMores can follow. Last chunk should be sent with replyLast (otherwise 200 ms wait for more data occurs).
    ## Returns `false` if socket was closed.
    discard

  template replyStart*(ctx: HttpCtx, code: HttpCode, contentlength: int, firstpart: string): bool =
    discard

  template replyMore*(ctx: HttpCtx, bodypart: string): bool =
    discard

  template replyLast*(ctx: HttpCtx, lastpart: string) =
    ## Ends a reply started with replyStart.
    discard

  template replyLast*(ctx: HttpCtx) =
    discard