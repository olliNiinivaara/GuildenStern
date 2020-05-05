import posix, net, nativesockets, os, strutils, streams
import ../guildenserver


{.push checks: off.}

proc setHeaderValue(c: GuildenVars, current: (string, string)): bool {.inline.} =
  let field = toLowerAscii(current[0])
  var i = 0
  while i <= c.gs.lastheader:
    if c.gs.headerfieldarray[i] == field:
      c.headervalues[i] = current[1]
      return true
    i.inc
  false


proc findFinishPos(c: GuildenVars, j: int, recvbufferlen: int) {.inline.} =
  #c.bodystartpos = recvbufferlen; return
  if j > recvbufferlen - 5: (c.bodystartpos = recvbufferlen; return)
  var i = j
  let contleni = c.gs.headerfieldarray.find("content-length")
  if contleni == -1: (c.bodystartpos = recvbufferlen; return)
  if c.recvbuffer.data[recvbufferlen-4] == '\c' and c.recvbuffer.data[recvbufferlen-3] == '\l' and c.recvbuffer.data[recvbufferlen-2] == '\c' and c.recvbuffer.data[recvbufferlen-1] == '\l':
      c.bodystartpos = recvbufferlen
      echo "finish löyty tutkimalla"
      return
  while i <= recvbufferlen - 4:
    if c.recvbuffer.data[i] == '\c' and c.recvbuffer.data[i+1] == '\l' and c.recvbuffer.data[i+2] == '\c' and c.recvbuffer.data[i+3] == '\l':
      c.bodystartpos = i + 4
      return
    i.inc


proc parseHeaders(c: GuildenVars, recvbufferlen: int): bool =
  c.bodystartpos = -1
  var headercount = 0
  for i in 0 .. c.gs.lastheader: c.headervalues[i].setLen(0)
  
  while c.methlen < recvbufferlen and c.recvbuffer.data[c.methlen] != ' ': c.methlen.inc
  if c.methlen == recvbufferlen: return false
  var i = c.methlen + 1
  let start = i
  while i < recvbufferlen and c.recvbuffer.data[i] != ' ': i.inc()
  c.path = start
  c.pathlen = i - start
  if c.gs.lastheader == 0:    
    findFinishPos(c, i+1, recvbufferlen)
    return c.bodystartpos > -1
  i.inc
  var value = false
  var current: (string, string) = ("", "")

  while i <= recvbufferlen - 4:
    case c.recvbuffer.data[i]
    of '\c':
      if c.recvbuffer.data[i+1] == '\l' and c.recvbuffer.data[i+2] == '\c' and c.recvbuffer.data[i+3] == '\l':
        c.bodystartpos = i + 4
        return true
    of ':':
      if value: current[1].add(':')
      value = true
    of ' ':
      if value:
        if current[1].len != 0: current[1].add(c.recvbuffer.data[i])
      else: current[0].add(c.recvbuffer.data[i])
    of '\l':
      if c.setHeaderValue(current): headercount.inc
      if headercount == c.gs.lastheader + 1:
        c.findFinishPos(i, recvbufferlen)
        return c.bodystartpos > -1
      value = false
      current = ("", "")
    else:
      if value: current[1].add(c.recvbuffer.data[i])
      else: current[0].add(c.recvbuffer.data[i])
    i.inc()
  return false

    
proc readFromHttp*(c: GuildenVars): bool {.gcsafe, raises:[] .} =
  var recvbufferlen = 0
  var trials = 0
  var expectedlength = MaxRequestLength + 1
  while true:
    let ret = recv(c.fd, addr c.recvbuffer.data[recvbufferlen], expectedlength - recvbufferlen, 0.cint)
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
      c.currentexceptionmsg = "http receive: " & $lastError & " " & osErrorMsg(OSErrorCode(lastError))
      return false

    recvbufferlen += ret
    if recvbufferlen == MaxRequestLength + 1:
      c.currentexceptionmsg = "http receive: Max request size exceeded"
      return false
    if recvbufferlen == expectedlength: break
    if expectedlength == MaxRequestLength + 1:
      try:
        let allheadersreceived = c.parseHeaders(recvbufferlen)
        if not allheadersreceived:
          trials += 1
          continue
        trials = 0
        let contlen = c.gs.headerfieldarray.find("content-length")
        if contlen == -1: break
        if c.headervalues[contlen] == "": break
        expectedlength = parseInt(c.headervalues[contlen]) + c.bodystartpos
        if recvbufferlen == expectedlength: break
      except:
        c.currentexceptionmsg = "parse http headers: " & getCurrentExceptionMsg()
        return false
  try:
    c.recvbuffer.setPosition(recvbufferlen) # - 1 ?
  except: echo("recvbuffer setPosition error")
  true

{.pop.}