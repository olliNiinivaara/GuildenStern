version       = "0.9.0"
author        = "Olli"
description   = "Modular multithreading Linux HTTP server"
license       = "MIT"
srcDir        = "src"

skipDirs = @[".github", "bench", "e2e-tests"]

requires "nim >= 1.4.2"

task test, "run all tests":
  exec "cd tests/test_basics/; nim c -r --gc:arc --threads:on --d:threadsafe test_basics.nim"
