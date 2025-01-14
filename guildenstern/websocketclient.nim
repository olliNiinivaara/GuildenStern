import std/[strutils, net, uri, httpclient, posix]
import guildenstern/[dispatcher, websocketserver]
export websocketserver

type
  WebsocketClient* {.inheritable.} = ref object
    id*: int
    url*: Uri
    httpclient*: HttpClient
    socket*: SocketHandle = INVALID_SOCKET
    clientele*: WebsocketClientele
    wsmessageCallback: proc(client: WebsocketClient) {.nimcall.}
    
  WebsocketClientele* = ref object of WebsocketServer
    clients*: seq[WebSocketClient]


var emptyClient = WebSocketClient(id: 0)


proc close*(client: WebsocketClient, handshake = true) =
  if client == emptyClient: return
  client.httpclient.close()
  client.clientele.closeSocket(client.socket)
  client.socket = INVALID_SOCKET


proc connect*(client: WebsocketClient, key = "dGhlIHNhbXBsZSBub25jZQ=="): bool =
  client.httpclient.headers = newHttpHeaders({ "Connection": "Upgrade", "Upgrade": "websocket", "Sec-WebSocket-Key": key, "Content-Length": "0"})
  try:
    discard client.httpclient.getContent(client.url)
  except:
    echo getCurrentExceptionMsg()
    return false
  client.socket = client.httpclient.getSocket().getFd()
  if not client.clientele.registerSocket(client.socket, 1, cast[pointer](client.id)):
    client.close()
    return false
  return true


proc clienteleReceive() {.nimcall, raises:[].} =
  {.gcsafe.}:
    try:
      var id = cast[int](socketcontext.customdata)
      let client = WebsocketClientele(wsserver).clients[id]
      if id > 0: client.wsmessageCallback(client)
    except:
      echo getCurrentExceptionMsg()


proc send*(client: WebSocketClient, message: string, timeoutsecs = 10): bool {.discardable.} =
  if client == emptyClient: return
  if not client.clientele.send(client.socket, message, false, timeoutsecs):
    client.close()
    return false
  return true


proc findClient*(clientele: WebsocketClientele, socket: SocketHandle): var WebSocketClient =
  for client in clientele.clients.mitems:
    if client.socket == socket: return client
  return emptyClient


proc isConnected*(client: WebSocketClient): bool =
  return client.socket != INVALID_SOCKET


iterator connectedClients*(clientele: WebsocketClientele): var WebsocketClient =
  for client in clientele.clients.mitems:
    if client.isConnected: yield(client)

 
proc newWebsocketClient*(clientele: WebsocketClientele, url: string,
 onwsmessagecallback: proc(client: WebsocketClient) {.nimcall.}): WebsocketClient =
  result = WebSocketClient(clientele: clientele, url: parseUri(url), wsmessageCallback: onwsmessagecallback)
  result.id = clientele.clients.len
  result.httpclient = newHttpClient()
  clientele.clients.add(result)
 

proc newWebsocketClientele*(onClosesocketcallback: OnCloseSocketCallback = nil, loglevel = LogLevel.WARN, bufferlength = 1000): WebsocketClientele =
  result = new WebsocketClientele
  initWebsocketServer(result, nil, nil, clienteleReceive, loglevel)
  result.onClosesocketcallback = onClosesocketcallback
  result.bufferlength = bufferlength
  result.clients.add(emptyClient)


proc start*(clientele: WebsocketClientele, threadpoolsize = 0): bool =
  return clientele.start(port = 0, threadpoolsize = threadpoolsize.uint)