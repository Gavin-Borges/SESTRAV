#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq g++ >/dev/null
export PATH="/mix:${PATH}"
cd /prime/lib
g++ -O3 -static PRIME.cc -o PRIME_docker
./PRIME_docker -i ../test/test.txt -o /tmp/prime_docker_out.txt -a A0201,A0101,A0301,A2402,A1101,B0702,B0801,B2705,B3501
wc -l /tmp/prime_docker_out.txt
head -3 /tmp/prime_docker_out.txt
