import posix, net, nativesockets, os, strutils, streams
import ../guildenserver, httphandlertype

const
  MaxRequestLength* {.intdefine.} = 1000
  MaxResponseLength* {.intdefine.} = 100000

var  
  headerfieldsArrays* : array[256, Headerfieldarray]
  lastheaders*: array[256, int]


{.push checks: off.}

proc setHeaderValue(h: HttpHandler, current: (string, string)): bool {.inline.} =
  let field = toLowerAscii(current[0])
  var i = 0
  while i <= lastheaders[h.gs.serverid]:
    {.gcsafe.}:
      if headerfieldsArrays[h.gs.serverid][i] == field:
        h.headervalues[i] = current[1]
        return true
    i.inc
  false


proc findFinishPos(h: HttpHandler, j: int, recvdatalen: int) {.inline.} =
  if j > recvdatalen - 5: (h.bodystartpos = recvdatalen; return)
  var i = j
  {.gcsafe.}:
    let contleni = headerfieldsArrays[h.gs.serverid].find("content-length")
    if contleni == -1: (h.bodystartpos = recvdatalen; return)
    if h.recvdata.data[recvdatalen-4] == '\c' and h.recvdata.data[recvdatalen-3] == '\l' and h.recvdata.data[recvdatalen-2] == '\c' and h.recvdata.data[recvdatalen-1] == '\l':
        h.bodystartpos = recvdatalen
        return
    while i <= recvdatalen - 4:
      if h.recvdata.data[i] == '\c' and h.recvdata.data[i+1] == '\l' and h.recvdata.data[i+2] == '\c' and h.recvdata.data[i+3] == '\l':
        h.bodystartpos = i + 4
        return
      i.inc


proc parseHeaders(h: HttpHandler, recvdatalen: int): bool =
  h.bodystartpos = -1
  var headercount = 0
  for i in 0 .. lastheaders[h.gs.serverid]: h.headervalues[i].setLen(0)
  
  while h.methlen < recvdatalen and h.recvdata.data[h.methlen] != ' ': h.methlen.inc
  if h.methlen == recvdatalen: return false
  var i = h.methlen + 1
  let start = i
  while i < recvdatalen and h.recvdata.data[i] != ' ': i.inc()
  h.path = start
  h.pathlen = i - start
  if lastheaders[h.gs.serverid] == 0:    
    findFinishPos(h, i+1, recvdatalen)
    return h.bodystartpos > -1
  i.inc
  var value = false
  var current: (string, string) = ("", "")

  while i <= recvdatalen - 4:
    case h.recvdata.data[i]
    of '\c':
      if h.recvdata.data[i+1] == '\l' and h.recvdata.data[i+2] == '\c' and h.recvdata.data[i+3] == '\l':
        h.bodystartpos = i + 4
        return true
    of ':':
      if value: current[1].add(':')
      value = true
    of ' ':
      if value:
        if current[1].len != 0: current[1].add(h.recvdata.data[i])
      else: current[0].add(h.recvdata.data[i])
    of '\l':
      if h.setHeaderValue(current): headercount.inc
      if headercount == lastheaders[h.gs.serverid] + 1:
        h.findFinishPos(i, recvdatalen)
        return h.bodystartpos > -1
      value = false
      current = ("", "")
    else:
      if value: current[1].add(h.recvdata.data[i])
      else: current[0].add(h.recvdata.data[i])
    i.inc()
  return false

    
proc readFromHttp*(h: HttpHandler): bool {.gcsafe, raises:[] .} =
  var recvdatalen = 0
  var trials = 0
  var expectedlength = MaxRequestLength + 1
  while true:
    let ret = recv(h.socket, addr h.recvdata.data[recvdatalen], expectedlength - recvdatalen, 0.cint)
    if h.gs.serverstate == Shuttingdown: return false
    if ret == 0:
      let lastError = osLastError().int    
      if lastError == 0: return recvdatalen > 0
      if lastError == 2 or lastError == 9 or lastError == 104: return false
      trials.inc
      if trials <= 3: 
        echo "nothing received, backoff triggered"
        sleep(100 + trials * 100)
        continue
        # TODO: real backoff strategy
    if ret == -1:
      let lastError = osLastError().int
      h.currentexceptionmsg = "http receive: " & $lastError & " " & osErrorMsg(OSErrorCode(lastError))
      return false
    if trials >= 4:
      h.currentexceptionmsg = "http receive: receiving stalled"
      return false
      
    recvdatalen += ret

    if recvdatalen == MaxRequestLength + 1:
      h.currentexceptionmsg = "http receive: Max request size exceeded"
      return false
    if recvdatalen == expectedlength: break
    if expectedlength == MaxRequestLength + 1:
      try:
        let allheadersreceived = h.parseHeaders(recvdatalen)
        if not allheadersreceived:
          trials += 1
          continue
        trials = 0
        {.gcsafe.}:
          let contlenpos = headerfieldsArrays[h.gs.serverid].find("content-length")
        if contlenpos == -1 or h.headervalues[contlenpos] == "": return true
        expectedlength = parseInt(h.headervalues[contlenpos]) + h.bodystartpos
        if recvdatalen == expectedlength: break
      except:
        h.currentexceptionmsg = "parse http headers: " & getCurrentExceptionMsg()
        return false
  try:
    h.recvdata.setPosition(recvdatalen) # - 1 ?
  except:
    echo("recvdata setPosition error")
    return false
  true

{.pop.}