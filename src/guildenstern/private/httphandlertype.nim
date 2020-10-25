# this file to avoid circular unit references

from streams import StringStream
import ../guildenserver

const
  MaxHttpHeaderFields* {.intdefine.} = 25

type
  Headerfieldarray* = array[MaxHttpHeaderFields, string]

  HttpHandler* = ref object of Handler
    recvdata* : StringStream
    senddata* : StringStream
    path*: int
    pathlen*: int
    methlen*: int
    headervalues* : Headerfieldarray
    bodystartpos* : int