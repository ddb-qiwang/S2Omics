#!/bin/bash
set -e

source_uni="https://upenn.box.com/s/n00mkwmlo0iej81dozti0fwwbsfhyhuv"
target_uni="checkpoints/uni_pytorch_model.bin"

mkdir -p checkpoints
wget ${source_uni} -O ${target_uni}
