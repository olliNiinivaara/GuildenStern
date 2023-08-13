import guildenstern/[dispatcher, guildenserver, httpserver, streamingserver, websocketserver]
import osproc
discard execCmd("nim doc --project --index:on --outdir:docs createdocs.nim")


# nim doc --project --index:on --outdir:../docs guildenserver.nim
# nim buildIndex -o:../docs/theindex.html ../docs
