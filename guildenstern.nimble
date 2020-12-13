version       = "0.9.0"
author        = "Olli"
description   = "Modular multithreading Linux HTTP server"
license       = "MIT"
srcDir        = "src"

skipDirs = @[".github", "bench"]

requires "nim >= 1.4.2"

task test, "run all tests":
  exec "cd tests/err_empty_response/; nim c -r --d:danger --gc:arc --threads:on --d:threadsafe test_err_empty_response.nim"
