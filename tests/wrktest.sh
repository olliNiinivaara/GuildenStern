# https://github.com/wg/wrk/wiki/Installing-Wrk-on-Linux

wrk -t5 -c5 -d10s --latency --timeout 10s http://127.0.0.1:8080/plaintext