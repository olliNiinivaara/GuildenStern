import guildenstern/[dispatcher, guildenserver, httpserver, multipartserver, websocketserver]
import osproc
discard execCmd("nim doc --project --index:on --outdir:docs createdocs.nim")

# rm docs/*
# cd guildenstern
# nim buildIndex -o:../docs/theindex.html ../docs
# nim doc --project --index:on --outdir:../docs guildenserver.nim etc until works
# rm *.idx

