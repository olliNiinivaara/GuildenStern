import cgi, guildenstern/[dispatcher, epolldispatcher, httpserver]
     
proc handleGet() =
  echo "method: ", getMethod()
  echo "uri: ", getUri()
  if isUri("/favicon.ico"): reply(Http204)
  else:
    reply """
      <!doctype html><title>GuildenStern Example</title><body>
      <form action="http://localhost:5051" method="post" accept-charset="utf-8">
      <input name="say" value="Hi"><button>Send"""

proc handlePost() =
  echo "client said: ", readData(getBody()).getOrDefault("say")
  reply(Http303, ["location: " & http.headers.getOrDefault("origin")])
  
let getserver = newHttpServer(handleGet, loglevel = lvlInfo, contenttype = NoBody)
let postserver = newHttpServer(handlePost, headerfields = ["origin"])
dispatcher.start(getserver, 5050)
epolldispatcher.start(postserver, 5051, threadpoolsize = 20)
joinThreads(getserver.thread, postserver.thread)