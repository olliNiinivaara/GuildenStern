import guildenstern/[dispatcher, httpserver]
import random

proc onRequest() =
  if isUri("/favicon.ico"): reply(Http204)
  elif isUri("/"):
    let html = """<!doctype html><title></title><body><a href="/download" download>Download</a>"""
    reply(html)
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
s.start(5050)
joinThread(s.thread)