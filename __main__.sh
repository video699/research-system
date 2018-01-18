#!/bin/sh
# This shell script implements the command-line interface.
PYTHONHASHSEED=12345 nice -n 19 python __main__.py "$@"
