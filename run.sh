#!/bin/bash

uv run main.py -m mlp -d 50 -s 1 -t random -n dense --log_level info "./pdtb/train.json" "./pdtb/dev.json" "./pdtb/test.json"
uv run main.py -m mlp -d 100 -s 1 -t random -n dense --log_level info "./pdtb/train.json" "./pdtb/dev.json" "./pdtb/test.json"
uv run main.py -m mlp -d 200 -s 1 -t random -n dense --log_level info "./pdtb/train.json" "./pdtb/dev.json" "./pdtb/test.json"

uv run main.py -m mlp -s 1 -n sparse --log_level info "./pdtb/train.json" "./pdtb/dev.json" "./pdtb/test.json"
