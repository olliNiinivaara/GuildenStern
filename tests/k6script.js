// install k6: https://k6.io/docs/getting-started/installation
// run: k6 run --vus 4 --iterations 4 k6script.js

import ws from "k6/ws"
import { check } from "k6"

export default function() {
  const url = "ws://localhost:8080"
  const requests = 100000
  let responses = 0

  const res = ws.connect(url, null, function(socket) {
    socket.on("open", function open() {
      console.log("Making " + requests + " requests")
      for (let x = 0; x < requests; x++) socket.send("PING!")
    })
    socket.on('message', function() {
      if (++responses == requests) socket.close()
    })
  })
  
  check(res, {
    "status is 101": r => r.status === 101,
    "every request received response": r => responses == requests
  })
}