# GuildenStern
Modular multithreading Linux HTTP server

## Documentation
http://htmlpreview.github.io/?https://github.com/olliNiinivaara/GuildenStern/blob/master/doc/guildenstern.html

## Installation

`nimble install https://github.com/olliNiinivaara/GuildenStern@#master`

## Features

- Modular architecture let's you add new handlers without forking the project
- Modularity means fewer bugs and more opportunities for performance optimization
- Can listen to multiple ports with different handlers
- Already supports streaming requests, streaming replies, and websocket
- Preemptive multithreading guarantees low latencies by fair access to CPU cores
- Supports --gc:arc, doesn't need asyncdispatch
- Runs in single-threaded mode, too