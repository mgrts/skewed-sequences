#!/usr/bin/env bash
# NVIDIA Multi-Process Service for the parallel sweep slots.
#   source scripts/mps.sh start    # start the daemon (as this user) AND export the pipe dir into this shell
#   bash   scripts/mps.sh status
#   bash   scripts/mps.sh stop
# Without MPS, several CUDA processes time-slice the GPU and tiny kernels make that far
# slower than running serially. The slots must be launched from a shell where
# CUDA_MPS_PIPE_DIRECTORY is exported (hence `source ... start`).
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-$HOME/.mps/pipe}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-$HOME/.mps/log}"
case "${1:-status}" in
  start)
    mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
    if echo get_server_list | nvidia-cuda-mps-control >/dev/null 2>&1; then echo "MPS daemon already running"
    else nvidia-cuda-mps-control -d && echo "MPS daemon started"; fi ;;
  stop)   echo quit | nvidia-cuda-mps-control && echo "MPS daemon stopped" ;;
  status) echo -n "MPS servers: "; echo get_server_list | nvidia-cuda-mps-control 2>/dev/null || echo "(daemon not running)" ;;
  *) echo "usage: source scripts/mps.sh start | bash scripts/mps.sh status|stop"; exit 1 ;;
esac
