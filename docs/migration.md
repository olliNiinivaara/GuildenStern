Some steps for upgrading a codebase to GuildenStern 6
====================================================

## Imports

Thanks to the new atlas package manager, I could get rid of the pointless intermediate *src* directory. This may require some path changes in your code.

Instead of importing ctxbody, ctxfull, ctxheader or ctxhttp, import **httpserver**

ctxstream -> **streamingserver**

ctxws -> **websocketserver**

ctxtimer does not exist anymore.

To start a server, import also **dispatcher**.

## Running a server

Create a new server with **newHttpServer**, **newStreamingServer** or **newWebsocketServer**.

**newWebsocketServer** takes as parameters four callback functions, where second (WsAfterUpgradeCallback) and last (OnCloseSocketCallback) can be set to nil.

Bind a server to a port and start **server.thread** with dispatcher's **start** proc. Note that this proc returns when
the thread is running, so you may want to let it run by calling **joinThread(s)(server.thread [, ...])**. There is currently no way to stop a single server, call **shutdown()** to bring them all down.

## Request context handlers

Previously, context data was brought to you with a ctx: HttpCtx parameter in every callback. This is now implicitly available as thread local parameter. For httpserver, it is called **http**, WebSocketServer has **ws**, and StreamingServer offers **stream**. The most important thing available in these is **http/ws/stream.socketdata.socket**. For accessing the server, there is a also convenience template that casts the http.socketdata.server into a HttpServer (works for streamingserver, too, because it is just a HttpServer with a custom SocketContext). For accessing websocketserver, there is **wsserver** template.

## Request for comments

That's it, basically. If you have further questions or answers, please send an issue or pull request, and more advice can be added. Happy web serving.








