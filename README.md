![Tests](https://github.com/olliNiinivaara/GuildenStern/workflows/Tests/badge.svg)

# GuildenStern
Modular multithreading Linux HTTP + WebSocket server

## Example

```nim
# nim r --d:release --d:threadsafe thisexample

import cgi, guildenstern/[dispatcher, httpserver]
     
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

[guildenserver](https://github.com/olliNiinivaara/GuildenStern/blob/dev/guildenstern/htmldocs/guildenserver.html)

## Installation

`git clone -b dev --single-branch https://github.com/olliNiinivaara/GuildenStern.git`

## Features

- Modular architecture: simple codebase and easy customization
- Every request is processed in dedicated thread: preemptive multitasking scales vertically
- Listen to different ports with different kinds of servers: creates opportunities for performance optimization

## Release notes, 6.0.0 (2023-07-29)

- major rewrite
