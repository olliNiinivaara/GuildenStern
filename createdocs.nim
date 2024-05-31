import guildenstern/[dispatcher, guildenserver, httpserver, multipartserver, websocketserver]
import osproc
discard execCmd("nim doc --project --index:on --outdir:docs createdocs.nim")

#[
# sha out?
rm docs/*
cd guildenstern
nim doc --project --index:on --outdir:../docs guildenserver.nim  # iterate with all until works
nim buildIndex -o:../docs/theindex.html ../docs
rm *.idx
]#

