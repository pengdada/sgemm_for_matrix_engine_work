#!/usr/bin/bash

repeate=100

./Release/cublas_sgemm 1 0 $repeate $((4096*4)) | tee log.txt
mkdir result_gpu
rm result_gpu/*
mv log.txt result_gpu
mv Power_data.txt result_gpu/power_data.gpu.csv


./Release/cublas_sgemm 1 1 $repeate $((4096*4)) | tee log.txt
mkdir result_tensorcore
rm result_tensorcore/*
mv log.txt result_tensorcore
mv Power_data.txt result_tensorcore/power_data.tensorcore.csv