const QueueSize {.intdefine.} = 500

type ThreadContext = tuple[gs: ptr GuildenServer, data: ptr SocketData]

var
  queue: array[QueueSize, ThreadContext]
  tail = 0
  head = -1
  workavailable: Cond
  worklock: Lock
  currentload, peakload, maxload: int
  threadinitializer: proc() {.nimcall, gcsafe, raises: [].}

initCond(workavailable)
initLock(worklock)

proc registerThreadInitializer*(callback: proc() {.nimcall, gcsafe, raises: [].}) = threadinitializer = callback

proc getLoads*(): (int, int, int) = (currentload, peakload, maxload)

proc resetMaxLoad*() = maxload = 0


proc threadProc() {.thread.} =
  if threadinitializer != nil: threadinitializer()
  currentload.atomicInc()
  while true:
    if unlikely(shuttingdown): break
    if unlikely(tail >= head):
      currentload.atomicDec()
      if currentload == 0: peakload = 0
      withLock(worklock): wait(workavailable, worklock)
      currentload.atomicInc()
      if currentload > peakload:
        peakload = currentload
        if peakload > maxload:
          maxload = peakload
      continue
    var mytail = tail.atomicInc()
    if unlikely(mytail > head): continue

    handleRead(queue[mytail].gs, queue[mytail].data)
    if likely(queue[mytail].data.port > 0) and likely(queue[mytail].gs.selector.contains(queue[mytail].data.socket.int)):
      try: queue[mytail].gs.selector.updateHandle(queue[mytail].data.socket.int, {Event.Read})
      except CatchableError:
        queue[mytail].gs[].log(ERROR, "updateHandle error: " & getCurrentExceptionMsg())
  

template createTask() =
  if unlikely(head == QueueSize - 1):
    var i = 0
    while tail < QueueSize - 1:
      if currentload < gs.workerthreadcount: signal(workavailable)
      i.inc
      if i == 1: discard sched_yield()
      elif i == 1000000: i = 0
      if shuttingdown: break
    tail = 0
    head = -1
    
  if unlikely(shuttingdown): break

  queue[head + 1].gs = unsafeAddr gs
  data.socket = fd
  queue[head + 1].data = data
  if tail > head: tail = head
  head.inc()
  if currentload < gs.workerthreadcount: signal(workavailable)