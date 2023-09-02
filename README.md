# GuildenStern
Modular multithreading Linux HTTP + WebSocket server

## Example

```nim
# nim r --d:release --d:threadsafe thisexample

import cgi, strtabs, guildenstern/[dispatcher, httpserver]
     
proc handleGet() =
  let html = """
    <!doctype html><title>GuildenStern Example</title><body>
    <form action="http://localhost:5051" method="post" accept-charset="utf-8">
    <input name="say" id="say" value="Hi"><button>Send"""
  reply(html)

proc handlePost() =
  try: echo readData(getBody()).getOrDefault("say")
  except: (reply(Http400) ; return)
  reply(Http303, ["location: http://localhost:5050"])

let getserver = newHttpServer(handleGet)
getserver.start(5050)
let postserver = newHttpServer(handlePost)
postserver.start(5051)
joinThreads(getserver.thread, postserver.thread)
```

## Documentation

https://olliniinivaara.github.io/GuildenStern/index.html

[migration guide](https://github.com/olliNiinivaara/GuildenStern/blob/master/docs/migration.md)

## Installation

atlas use GuildenStern


## Release notes, 6.1.0 (2023-09-02)

- fixed nimble path
- fixed README example
- new receiveInChucks iterator in streamingserver for receiving big data, for example POST data that does not fit in main memory
- new startDownload-continueDownload-finishDownload combo in streamingserver for sending big and/or dynamic responses as *Transfer-Encoding: chunked*
- new and upgraded streamingserver examples as required

