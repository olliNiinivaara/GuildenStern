# nim c -r --d:danger --threads:on --d:threadsafe tester.nim

import asyncdispatch, httpclient
from guildenstern import signalSIGINT
import guildentest
from os import sleep

proc testing() {.async.} =
  let c1 = newAsyncHttpClient()
  let c2 = newAsyncHttpClient()
  let c3 = newAsyncHttpClient()
  let c4 = newAsyncHttpClient()
  
  let r1 = c1.get("http://localhost:8080/plaintext")
  let r2 = c2.get("http://localhost:8080/plaintext")
  let r3 = c3.get("http://localhost:8080/plaintext")
  let r4 = c4.get("http://localhost:8080/plaintext")
  
  let b3 = await r3
  let b2 = await r2
  let b4 = await r4
  let b1 = await r1

  echo "b1 ", await b1.body
  echo "b4 ", await b4.body
  echo "b2 ", await b2.body
  echo "b3 ", await b3.body

var serverthread: Thread[void]
createThread(serverthread, startServer8080)

proc startClient() = waitFor testing()

var clientthread: Thread[void]
sleep(1000)
createThread(clientthread, startClient)

joinThread clientthread
signalSIGINT()