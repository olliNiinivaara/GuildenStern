import std/[strutils, net, os, uri, posix]
import guildenstern/[websocketserver]
export websocketserver

type
  WsClientMessageCallback* = proc(client: WsClient)
  WsClient* = ref object
    id*: int
    connected*: bool
    url*: Uri
    socket*: SocketHandle
    flags*: int
    messagecallback: WsClientMessageCallback
    headers: HttpHeaders
    sendtimeoutsecs: int
    socketobject: Socket
    socketserver: WebsocketServer
    readthread: Thread[WsClient]

proc setTheFlags(server: GuildenServer, socket: posix.SocketHandle, flags: int): bool = return true

proc suspend(serverid: int, sleepmillisecs: int) {.gcsafe, nimcall, raises: [].} = sleep(sleepmillisecs)

proc closeSocketImpl(server: GuildenServer, socket: SocketHandle, cause: SocketCloseCause, msg: string) {.gcsafe, nimcall, raises: [].} = discard


proc newWsClient(url: string, onwsmessagecallback: WsClientMessageCallback, recvtimeoutsecs = 5, sendtimeoutsecs = 5): WsClient =
  result = new(WsClient)
  result.url = parseUri(url)
  result.messagecallback = onwsmessagecallback
  result.socket = INVALID_SOCKET
  result.headers = newHttpHeaders()
  result.sendtimeoutsecs = sendtimeoutsecs
  result.socketserver = newWebsocketServer(nil, nil, nil, nil, INFO)
  result.socketserver.closeSocketCallback = closeSocketImpl
  result.socketserver.suspendCallback = suspend
  result.socketserver.setFlagsCallback = setTheFlags
  result.socketserver.sockettimeoutms = recvtimeoutsecs * 1000
  result.socketserver.bufferlength = 500
  result.socketserver.isclient = true


proc close*(client: WsClient) =
  if client.connected and client.socket != INVALID_SOCKET:
    discard pthread_cancel(Pthread(client.readthread.handle()))
    client.socketserver.sendClose(client.socket)
    discard client.socket.close()
  client.socket = INVALID_SOCKET
  client.connected = false
  

proc newConnection(client: WsClient) =
    if client.connected:
      client.close()
      client.connected = false
    let port = Port(client.url.port.parseInt)
    client.socketobject = net.dial(client.url.hostname, port)
    client.socket = client.socketobject.getFd()
    client.connected = true


proc generateHeaders(requestUrl: Uri, httpMethod: HttpMethod, headers: HttpHeaders): string =
  result = $httpMethod
  result.add ' '
  result.add($requestUrl)
  result.add(" HTTP/1.1" & httpNewLine)
  if not headers.hasKey("Connection"): add(result, "Connection: Keep-Alive" & httpNewLine)
  for key, val in headers: add(result, key & ": " & val & httpNewLine)
  add(result, httpNewLine)


proc readLoop(client: WsClient) =
  {.gcsafe.}:
    client.socketserver.handleWsThreadInitialization()
    var buffer = newString(1)
    while true:
      if not client.connected: break
      if shuttingdown: break
      if client.socket == INVALID_SOCKET: break
      let ret = recv(client.socket, addr buffer[0], 1, MSG_PEEK)
      if ret == 1:
        socketcontext.server = client.socketserver
        socketcontext.socket = client.socket
        handleWsRequest()
        client.messagecallback(client)
      else:
        if not shuttingdown: echo "socket ", client.socket, " failed"
        client.close()
        break

proc notUpgraded(server: GuildenServer, socket: posix.SocketHandle): int = 0
proc upgraded(server: GuildenServer, socket: posix.SocketHandle): int = 1
 
proc connect*(url: string, onwsmessagecallback: WsClientMessageCallback, timeoutsecs = 5): WsClient =
  let key = "dGhlIHNhbXBsZSBub25jZQ=="
  let client = newWsClient(url, onwsmessagecallback, timeoutsecs)
  client.headers = newHttpHeaders({ "Connection": "Upgrade", "Upgrade": "websocket", "Sec-WebSocket-Key": key, "Content-Length": "0"})
  client.socketserver.handleWsThreadInitialization()
  newConnection(client)
  let headerString = generateHeaders(client.url, HttpGet, client.headers)
  client.socketserver.getFlagsCallback = notUpgraded
  client.socketobject.send(headerString)
  var buffer = newString(1)
  let ret = recv(client.socket, addr buffer[0], 1, MSG_PEEK)
  if ret == 1:
    socketcontext.server = client.socketserver
    socketcontext.socket = client.socket
    handleWsRequest()
    client.socketserver.getFlagsCallback = upgraded
  else:
    echo "websocket handshake failed for socket ", client.socket
    client.close()
    return
  createThread(client.readthread, readLoop, client)
  return client


proc send*(client: WsClient, message: string, binary = false): bool =
  if not client.connected: return false
  return client.socketserver.send(client.socket, message, binary, client.sendtimeoutsecs)
  
