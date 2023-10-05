#!/bin/bash
set -e -x

gcc -c -fPIC sys_perf.c -I/usr/include/python3.6
ld -o sys_perf.so -shared sys_perf.o -lpython3.6m
rm -rf sys_perf.o

# Test
sudo python3 PyPerf.py