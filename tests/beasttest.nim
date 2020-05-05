# nimble install httpbeast
# nim c -r --d:danger --threads:on --d:threadsafe beasttest.nim

import options, asyncdispatch
import httpbeast
# import bcrypt

proc onRequest(req: Request): Future[void] =
  if req.httpMethod == some(HttpGet):
    case req.path.get()
    of "/plaintext":
      const data = "Hello, World!"
      let headers = "Content-Type: text/plain\r\nCookie: " # & hash("cookie", genSalt(8))
      req.send(Http200, data, headers)
    else:
      req.send(Http404)

run(onRequest)