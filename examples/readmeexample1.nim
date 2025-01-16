import guildenstern/[dispatcher, httpserver]
let server = newHttpServer(proc() = reply "hello world")
if server.start(8080): joinThread(server.thread)