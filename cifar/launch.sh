#!/bin/bash

# This script launches test.py on 4 GPUs via torchrun inside a detached tmux session,
# logging all outputs (stdout+stderr) to terminal.txt.

tmux new-session -d -s cfl_session \
  "torchrun --nproc_per_node=4 test.py 2>&1 | tee terminal.txt"
