# PATH=$LOCAL_HOME/cuda-10.0/bin:$PATH nvprof --normalized-time-unit ms --print-gpu-trace ./Release/dmove > /dev/null 2>nvcc-results
name=res.nvvp
rm -rf $name
PATH=$LOCAL_HOME/cuda-10.0/bin:$PATH nvprof --metrics all -o $name ./Release/cublas_sgemm 2 1 1 16384
