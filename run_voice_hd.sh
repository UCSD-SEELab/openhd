#!/bin/bash
#OPENHD_EXECUTION=./clean_run.sh
OPENHD_EXECUTION=python

${OPENHD_EXECUTION} examples/voicehd.py -t examples/dataset/isolet_train.choir_dat -i examples/dataset/isolet_test.choir_dat

