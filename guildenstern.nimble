version       = "0.9.0"
author        = "Olli"
description   = "Modular multithreading Linux HTTP server"
license       = "MIT"
srcDir        = "src"

skipDirs = @[".github", "bench"]

requires "nim >= 1.4.2"

task test, "run all tests":
  exec "nim c -r --d:danger --gc:arc --threads:on --d:threadsafe tests/err_empty_response/test_err_empty_response.nim"
