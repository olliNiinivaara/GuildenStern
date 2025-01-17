import guildenstern/[dispatcher, httpserver]
import random

proc onRequest() =
  if isUri("/favicon.ico"): reply(Http204)
  elif isUri("/"):
    reply """<!doctype html><title></title><body><a href="/download" download>Download</a>"""
  else:
    if not replyStartChunked(): (shutdown() ; return)
    var sentchunks = 0
    while sentchunks < 5:
      var chunk: string
      for _ in 1..rand(1 .. 1000): chunk.add(char(rand(int('A') .. int('Z'))))
      if not replyContinueChunked(chunk): break
      sentchunks += 1
    replyFinishChunked()

let s = newHttpServer(onRequest)
if not s.start(5050): quit()
joinThread(s.thread)