#!/usr/bin/env bash
echo '' > output.txt
make run ARGS="-n 128" >> output.txt