# nim c -r --d:danger --threads:on --d:threadsafe -d:WEAVE_NUM_THREADS=4 tester.nim

import asyncdispatch, httpclient

import guildentest

proc tests() {.async.} =
  let client = newAsyncHttpClient()
  let resp = await client.get("http://localhost:8080/plaintext")
  doAssert resp.code == Http200
  let body = await resp.body
  doAssert body == "Hello, World!"
  echo body


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
createThread(serverthread, startServer)

proc startClient() =
  waitFor testing()

var clientthread: Thread[void]
createThread(clientthread, startClient)

joinThread clientthread
stopServer()