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
  if not client.clientele.registerSocket(client.socket, cast[pointer](client.id)):
    client.close()
    return false
  if not client.clientele.setFlags(client.socket, 1): return false
  return true


proc clienteleReceive() {.nimcall, raises:[].} =
  {.gcsafe.}:
    try:
      var id = cast[int](socketcontext.customdata)
      let client = WebsocketClientele(wsserver).clients[id]
      if id > 0: client.wsmessageCallback(client)
    except:
      echo getCurrentExceptionMsg()


proc send*(client: WebSocketClient, message: string): bool {.discardable.} =
  if not client.clientele.send(client.socket, message):
    client.close()
    return false
  return true


proc isConnected*(client: WebSocketClient): bool =
  return client.socket != INVALID_SOCKET


iterator connectedClients*(clientele: WebsocketClientele): var WebsocketClient =
  for client in clientele.clients.mitems:
    if client.isConnected: yield(client)

 
proc newWebsocketClient*(clientele: WebsocketClientele, url: string,
 onwsmessagecallback: proc(client: WebsocketClient) {.nimcall.}): WebsocketClient =
  result = WebSocketClient(clientele: clientele, url: parseUri(url), wsmessageCallback: onwsmessagecallback)
  result.httpclient = newHttpClient()
  result.id = clientele.clients.len
  clientele.clients.add(result)


proc newWebsocketClientele*(onClosesocketcallback: OnCloseSocketCallback = nil, loglevel = LogLevel.WARN): WebsocketClientele =
  result = new WebsocketClientele
  initWebsocketServer(result, nil, nil, clienteleReceive, loglevel)
  result.onClosesocketcallback = onClosesocketcallback
  result.clients.add(emptyClient)


proc start*(clientele: WebsocketClientele): bool =
  return clientele.start(0)