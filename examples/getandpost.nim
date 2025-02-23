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
  
let getserver = newHttpServer(handleGet, loglevel = INFO, contenttype = NoBody)
let postserver = newHttpServer(handlePost, headerfields = ["origin"])
if not dispatcher.start(getserver, 5050): quit()
if not epolldispatcher.start(postserver, 5051, threadpoolsize = 20): quit()
joinThreads(getserver.thread, postserver.thread)