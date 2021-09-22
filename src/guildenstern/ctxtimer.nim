from selectors import registerTimer, getData, unregister

when not defined(nimdoc):
  import guildenstern
  export guildenstern
else:
  import guildenserver

type TimerDatum = object
  fd: int
  timerhandler: proc() {.nimcall, gcsafe, raises: [].}

var
  timerdata: array[MaxHandlersPerCtx, TimerDatum]

{.push checks: off.}

proc at(fd: int): int {.inline.} =
  while timerdata[result].fd != fd: result += 1

proc handleTimer(gs: ptr GuildenServer, data: ptr SocketData) {.gcsafe, nimcall, raises: [].} =
  {.gcsafe.}: timerdata[at(int(data.socket))].timerhandler()

proc initTimerCtx*(gs: var GuildenServer, interval: int, timercallback: proc() {.nimcall, gcsafe, raises: [].}) =
  ## Initializes a timer handler with given callback proc.
  var index = 0
  while index < MaxHandlersPerCtx and timerdata[index].fd != 0: index += 1
  if index == MaxHandlersPerCtx: raise newException(Exception, "Cannot register over " & $MaxHandlersPerCtx & " handlers per TimerCtx")
  let ctxid = gs.registerHandler(handleTimer, 0, "timer")
  timerdata[index] = TimerDatum(fd: gs.selector.registerTimer(interval, false, SocketData(ctxid: ctxid)), timerhandler: timercallback)


proc registerTimerhandler*(gs: var GuildenServer, callback: proc() {.nimcall, gcsafe, raises: [].}, interval: int) {.deprecated: "use initTimerCtx instead".} =
  gs.initTimerCtx(interval, callback)


proc removeTimerCtx*(gs: GuildenServer, timercallback: proc() {.nimcall.}) {.raises: [].} =
  ## Removes timer handler with given callback proc.
  var index = 0
  while index < MaxHandlersPerCtx and timerdata[index].timerhandler != timercallback: index += 1
  if index == MaxHandlersPerCtx:
    gs.log(ERROR, "could not remove nonexistent timer callback")
    return
  try: gs.selector.unregister(timerdata[index].fd)
  except: discard
  timerdata[index].fd = 0
{.pop.}