#   Guildenstern
#
#  Modular multithreading Linux HTTP server
#
#  (c) Copyright 2020 Olli Niinivaara
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#  
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
## A modular multithreading Linux HTTP server.
## Easily create and add handlers.
## Associate handlers with ports.
## Genuinely multithreading: spawns new thread for every request.
## Runs in single-threaded mode, too.
## No need to use async/await.
## 
## Example
## =======
## 
## In this example port number is coded to html just for demonstration purposes. In reality, use your
## reverse proxy to route different types of requests to different ports.
## 
## .. code-block:: Nim
##
##    # nim c -r --gc:arc --d:release --threads:on --d:threadsafe example.nim
##    
##    import cgi, guildenstern/[ctxheader, ctxfull]
##       
##    let origin = "http://localhost:5050"
##    let html = """
##    <!doctype html><title>GuildenStern Example</title><body>
##    <form action="http://localhost:5051" method="post">
##    <input name="say" id="say" value="Hi"><button>Send"""
##      
##    proc handleGet(ctx: HttpCtx) = ctx.reply(Http200, html)
##    
##    proc handlePost(ctx: HttpCtx, headers: StringTableRef) =
##      try: echo readData(ctx.getBody()).getOrDefault("say")
##      except: discard
##      ctx.reply(Http303, ["location: " & origin])
##          
##    var server = new GuildenServer
##    server.initHeaderCtx(handleGet, 5050)
##    server.initFullCtx(handlePost, 5051)
##    echo "GuildenStern HTTP server serving at ", origin
##    server.serve()
## 
## See also
## ========
## 
## | `ctxheader <http://htmlpreview.github.io/?https://github.com/olliNiinivaara/GuildenStern/blob/master/doc/ctxheader.html>`_
## | `ctxfull <http://htmlpreview.github.io/?https://github.com/olliNiinivaara/GuildenStern/blob/master/doc/ctxfull.html>`_
## | `ctxstream <http://htmlpreview.github.io/?https://github.com/olliNiinivaara/GuildenStern/blob/master/doc/ctxstream.html>`_
## | `ctxws <http://htmlpreview.github.io/?https://github.com/olliNiinivaara/GuildenStern/blob/master/doc/ctxws.html>`_
##


when not defined(nimdoc):
  import guildenstern/guildenserver
  export guildenserver
  import guildenstern/dispatcher
  export serve  
  import guildenstern/ctxhttp
  export ctxhttp
else:
  import strtabs, httpcore
  from nativesockets import SocketHandle
  const
    MaxHeaderLength* {.intdefine.} = 10000
      ## Maximum acceptable character length for HTTP header.
    MaxRequestLength* {.intdefine.} = 100000
      ## Maximum acceptable length (header length + content length) for a HTTP request
      ## (Note: in streaming handlers, this denotes chunk size).
  
 
  type
    CtxId = distinct int

    GuildenServer* {.inheritable.} = ref object
      ## | This is available in request context as ``gs`` property.
      ## | You can inherit this and add custom properties as needed.
   
    SocketData* = object
      ## | Data associated with a request context.
      ## | This is available in request context as ``socketdata`` property (probably not needed in application development, though).
      port*: uint16
      socket*: nativesockets.SocketHandle
      ctxid*: CtxId
      customdata*: pointer

    Ctx* {.inheritable, shallow.} = ref object
      ## | Common abstract base class for all request contexts.
      ## | A request context is received as a parameter in request callbacks. 
      gs*: ptr GuildenServer
      socketdata*: ptr SocketData

    HttpCtx* = ref object of Ctx
      ## | Common abstract base class for all HTTP handling request contexts.

    RequestCallback* = proc(ctx: Ctx) {.nimcall, raises: [].}
      ## Type for procs that are called when a socket request can be processed.
      ## When multithreading, serialize mutations to global vars with locks (use threadvars instead unless memory usage is a problem).
       
    TimerCallback* = proc() {.raises: [].}
   
    ThreadInitializationCallback* = proc() {.gcsafe, raises: [].}

    LostCallback* = proc(gs: ptr GuildenServer, data: ptr SocketData, lostsocket: SocketHandle){.raises: [].}
    
    ErrorCallback* = proc(msg: string) {.gcsafe, raises: [].}
 
  proc serve*(gs: GuildenServer, multithreaded = true) {.gcsafe.} =
    ## Starts server event dispatcher loop which runs until shutdown() is called or SIGINT received.
    ## If you want the server to run in a single thread, set multithreaded to false.
    ## 
    ## **Example:**
    ##
    ## .. code-block:: Nim
    ##
    ##    import guildenstern/ctxheader  
    ##    proc onRequest(ctx: HttpCtx) = ctx.reply(Http200)
    ##    var server = new GuildenServer
    ##    server.initHeaderCtx(onRequest, 5050)
    ##    server.serve(false)
    discard

  proc registerThreadInitializer*(gs: GuildenServer, callback: ThreadInitializationCallback) =
    ## Registers a procedure in which you can initialize threadvars. This is called exactly once for each thread, when it is first created.
    ## 
    ## **Example:**
    ##
    ## .. code-block:: Nim
    ##
    ##    import guildenstern, os, random
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
    ##    var server = new GuildenServer
    ##    server.registerThreadInitializer(initializeThreadvars)
    ##    server.registerTimerhandler(tiktok, 100)
    ##    server.serve()
    ##    echo "----"
    ##    sleep(2510)
    discard

  proc registerTimerhandler*(gs: GuildenServer, callback: TimerCallback, interval: int) =
    ## Registers a new timer that fires the TimerCallback every `interval` milliseconds.
    discard

  proc registerConnectionlosthandler*(gs: GuildenServer, callback: LostCallback) =
    ## Registers procedure that is called when socket connection was dropped.
    ## This is not called when closeSocket() is called.
    discard

  proc registerErrornotifier*(gs: GuildenServer, callback: ErrorCallback) =
    ## Registers a proc that gets called when an internal error occurs.
    ## Note that you can get even more debug info by compiling with switch -d:fulldebug.
    discard

  proc closeSocket*(ctx: Ctx) {.raises: [].} =
    ## Closes the socket associated with the request context.
    discard

  proc closeSocket*(gs: ptr GuildenServer, socket: nativesockets.SocketHandle) {.raises: [].} =
    ## Closes the socket.
  
  proc shutdown*() =
    ## Cancels pending network I/O and breaks event dispatcher loops of all servers. Sending SIGINT / pressing Ctrl-C will automatically call shutdown().
    ## 
    ## **Example:**
    ##
    ## .. code-block:: Nim
    ##
    ##   import guildenstern, threadpool, os
    ##   
    ##   echo "Hit Ctrl-C or wait until B hits 10..."
    ##    
    ##   var a: int ; var serverA = new GuildenServer
    ##   serverA.registerTimerhandler((proc() =
    ##     a += 1 ;  let aa = a; echo "A ", aa, "->" ; sleep(2000) ; echo "<-A ", aa), 600)
    ##   spawn serverA.serve()
    ##   
    ##   var b: int ; var serverB = new GuildenServer
    ##   serverB.registerTimerhandler((proc() =
    ##     b += 1 ; let bb = b; echo "B ", bb, "->" ; sleep(2000) ; echo "<-B ", bb
    ##     if b == 10: b += 1; echo "commencing shutdown."; shutdown()), 750)
    ##   serverB.serve()
    ##   
    ##   echo "waiting for processes to finish..."
    ##   sleep(2010)
    ##   echo "graceful shutdown handling here!"
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
    ##    var server = new GuildenServer
    ##    server.registerThreadInitializer(initializeThreadvars)
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