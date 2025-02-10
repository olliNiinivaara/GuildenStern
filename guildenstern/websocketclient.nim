## This module allows easy creation of lots of websocket clients for testing purposes.
##
## The WebsocketClientele object manages the operation and holds the set of all created
## clients in the `clients` seq parameter. (Note that the client at index position zero
## is always void client with id zero).
##
## The individual clients are represented by the WebsocketClient object, which is inheritable.

import std/[strutils, net, uri, posix]
import websocketserver
export websocketserver
when not defined(nimdoc): import std/httpclient

when epollSupported(): import epolldispatcher
else: import dispatcher

when not defined(nimdoc):
  type
    WebsocketClient* {.inheritable.} = ref object
      id*: int
      url*: Uri
      socket*: SocketHandle = INVALID_SOCKET
      httpclient*: HttpClient
      clientele*: WebsocketClientele
      wsmessageCallback: proc(client: WebsocketClient) {.nimcall.}
    
    WebsocketClienteleObj* = object of WebsocketServerObj
      clients*: seq[WebSocketClient]
    
    WebsocketClientele* = ptr WebsocketClienteleObj
else:
  type
    WebsocketClient* {.inheritable.} = ref object
      id*: int
      url*: Uri
      socket*: SocketHandle = INVALID_SOCKET
      clientele*: WebsocketClientele
      wsmessageCallback: proc(client: WebsocketClient) {.nimcall.}

    WebsocketClienteleObj* = object of WebsocketServerObj
      clients*: seq[WebSocketClient]
    
    WebsocketClientele* = ptr WebsocketClienteleObj


var emptyClient = WebSocketClient(id: 0)


proc close*(client: WebsocketClient, handshake = true) =
  ## if you want to bypass the websocket close handshake dance, set the `handhake` to false.
  if client == emptyClient: return
  when not defined(nimdoc): client.httpclient.close()
  client.clientele.closeSocket(client.socket)
  client.socket = INVALID_SOCKET


proc connect*(client: WebsocketClient, key = "dGhlIHNhbXBsZSBub25jZQ=="): bool =
  ## Tries to connect the client to a server. According to the spec, the key should be something random.
  when not defined(nimdoc):
    client.httpclient.headers = newHttpHeaders({ "Connection": "Upgrade", "Upgrade": "websocket", "Sec-WebSocket-Key": key, "Sec-WebSocket-Version": "13", "Content-Length": "0"})
  try:
    when not defined(nimdoc):
      discard client.httpclient.getContent(client.url)
    else: discard
  except:
    echo getCurrentExceptionMsg()
    return false
  when not defined(nimdoc):
    client.socket = client.httpclient.getSocket().getFd()
  if not client.clientele.registerSocket(client.socket, 1, cast[pointer](client.id)):
    client.close()
    return false
  return true


proc clienteleReceive() {.nimcall, raises:[].} =
  {.gcsafe.}:
    try:
      var id = cast[int](socketcontext.customdata)
      let client = cast[WebsocketClientele](wsserver).clients[id]
      if id > 0: client.wsmessageCallback(client)
    except:
      echo getCurrentExceptionMsg()


proc send*(client: WebSocketClient, message: string, timeoutsecs = 10): bool {.discardable.} =
  ## If sending fails, closes the client automatically
  if client == emptyClient: return
  if not client.clientele.send(client.socket, message, false, timeoutsecs):
    client.close()
    return false
  return true


proc findClient*(clientele: WebsocketClientele, socket: SocketHandle): WebSocketClient =
  ## If such socket is not found, returns the empty client (client with id zero) 
  for client in clientele.clients.items:
    if client.socket == socket: return client
  return emptyClient


proc isConnected*(client: WebSocketClient): bool =
  return client.socket != INVALID_SOCKET


iterator connectedClients*(clientele: WebsocketClientele): WebsocketClient =
  for client in clientele.clients.items:
    if client.isConnected: yield(client)

 
proc newWebsocketClient*(clientele: WebsocketClientele, url: string,
 receive: proc(client: WebsocketClient) {.nimcall.}): WebsocketClient =
  result = WebSocketClient(clientele: clientele, url: parseUri(url), wsmessageCallback: receive)
  result.id = clientele.clients.len
  when not defined(nimdoc): result.httpclient = newHttpClient()
  clientele.clients.add(result)
 

proc newWebsocketClientele*(close: OnCloseSocketCallback = nil, loglevel = LogLevel.WARN, bufferlength = 1000, bytemask = "\11\22\33\44"): WebsocketClientele =
  ## Makes sense to keep the bufferlength low, if you are running thousands of clients.
  ## According to the spec, the bytemask should be random. 
  result = cast[WebsocketClientele](allocShared0(sizeof(WebsocketClienteleObj)))
  initWebsocketServer(result, nil, nil, clienteleReceive, loglevel, bytemask)
  result.onClosesocketcallback = close
  result.bufferlength = bufferlength
  result.clients.add(emptyClient)


proc start*(clientele: WebsocketClientele, threadpoolsize = 0): bool =
  return clientele.start(port = 0, threadpoolsize = threadpoolsize.uint)