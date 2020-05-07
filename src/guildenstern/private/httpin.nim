import posix, net, nativesockets, os, strutils, streams
import ../guildenserver


{.push checks: off.}

proc setHeaderValue(gv: GuildenVars, current: (string, string)): bool {.inline.} =
  let field = toLowerAscii(current[0])
  var i = 0
  while i <= gv.gs.lastheader:
    if gv.gs.headerfieldarray[i] == field:
      gv.headervalues[i] = current[1]
      return true
    i.inc
  false


proc findFinishPos(gv: GuildenVars, j: int, recvbufferlen: int) {.inline.} =
  if j > recvbufferlen - 5: (gv.bodystartpos = recvbufferlen; return)
  var i = j
  let contleni = gv.gs.headerfieldarray.find("content-length")
  if contleni == -1: (gv.bodystartpos = recvbufferlen; return)
  if gv.recvbuffer.data[recvbufferlen-4] == '\c' and gv.recvbuffer.data[recvbufferlen-3] == '\l' and gv.recvbuffer.data[recvbufferlen-2] == '\c' and gv.recvbuffer.data[recvbufferlen-1] == '\l':
      gv.bodystartpos = recvbufferlen
      return
  while i <= recvbufferlen - 4:
    if gv.recvbuffer.data[i] == '\c' and gv.recvbuffer.data[i+1] == '\l' and gv.recvbuffer.data[i+2] == '\c' and gv.recvbuffer.data[i+3] == '\l':
      gv.bodystartpos = i + 4
      return
    i.inc


proc parseHeaders(gv: GuildenVars, recvbufferlen: int): bool =
  gv.bodystartpos = -1
  var headercount = 0
  for i in 0 .. gv.gs.lastheader: gv.headervalues[i].setLen(0)
  
  while gv.methlen < recvbufferlen and gv.recvbuffer.data[gv.methlen] != ' ': gv.methlen.inc
  if gv.methlen == recvbufferlen: return false
  var i = gv.methlen + 1
  let start = i
  while i < recvbufferlen and gv.recvbuffer.data[i] != ' ': i.inc()
  gv.path = start
  gv.pathlen = i - start
  if gv.gs.lastheader == 0:    
    findFinishPos(gv, i+1, recvbufferlen)
    return gv.bodystartpos > -1
  i.inc
  var value = false
  var current: (string, string) = ("", "")

  while i <= recvbufferlen - 4:
    case gv.recvbuffer.data[i]
    of '\c':
      if gv.recvbuffer.data[i+1] == '\l' and gv.recvbuffer.data[i+2] == '\c' and gv.recvbuffer.data[i+3] == '\l':
        gv.bodystartpos = i + 4
        return true
    of ':':
      if value: current[1].add(':')
      value = true
    of ' ':
      if value:
        if current[1].len != 0: current[1].add(gv.recvbuffer.data[i])
      else: current[0].add(gv.recvbuffer.data[i])
    of '\l':
      if gv.setHeaderValue(current): headercount.inc
      if headercount == gv.gs.lastheader + 1:
        gv.findFinishPos(i, recvbufferlen)
        return gv.bodystartpos > -1
      value = false
      current = ("", "")
    else:
      if value: current[1].add(gv.recvbuffer.data[i])
      else: current[0].add(gv.recvbuffer.data[i])
    i.inc()
  return false

    
proc readFromHttp*(gv: GuildenVars): bool {.gcsafe, raises:[] .} =
  var recvbufferlen = 0
  var trials = 0
  var expectedlength = MaxRequestLength + 1
  while true:
    let ret = recv(gv.fd, addr gv.recvbuffer.data[recvbufferlen], expectedlength - recvbufferlen, 0.cint)
    if ret == 0:
      let lastError = osLastError().int    
      if lastError == 0 and recvbufferlen > 0: return true
      if lastError == 0 or lastError == 2 or lastError == 9 or lastError == 104: return false
      trials.inc
      if trials < 6: 
        echo "nothing received, backoff triggered"
        sleep(100 + trials * 100)
        continue
        # TODO: real backoff strategy
    if trials > 5 or ret == -1:
      let lastError = osLastError().int
      if lastError == 0 or lastError == 2 or lastError == 9 or lastError == 104: return false 
      gv.currentexceptionmsg = "http receive: " & $lastError & " " & osErrorMsg(OSErrorCode(lastError))
      return false

    recvbufferlen += ret
    if recvbufferlen == MaxRequestLength + 1:
      gv.currentexceptionmsg = "http receive: Max request size exceeded"
      return false
    if recvbufferlen == expectedlength: break
    if expectedlength == MaxRequestLength + 1:
      try:
        let allheadersreceived = gv.parseHeaders(recvbufferlen)
        if not allheadersreceived:
          trials += 1
          continue
        trials = 0
        let contlen = gv.gs.headerfieldarray.find("content-length")
        if contlen == -1: break
        if gv.headervalues[contlen] == "": break
        expectedlength = parseInt(gv.headervalues[contlen]) + gv.bodystartpos
        if recvbufferlen == expectedlength: break
      except:
        gv.currentexceptionmsg = "parse http headers: " & getCurrentExceptionMsg()
        return false
  try:
    gv.recvbuffer.setPosition(recvbufferlen) # - 1 ?
  except: echo("recvbuffer setPosition error")
  true

{.pop.}