#!/usr/bin/bash

repeate=100

function gemm()
{
	dtype=$1
	size=$2

	cmd="./Release/cublas_sgemm 1 0 $repeate $dtype $size | tee log.txt"
	echo $cmd; eval $cmd;
	dir=result.gpu.$dtype.$size
	mkdir -p $dir
	rm -f $dir/*
	mv log.txt $dir/log.$dtype.$size.txt
	mv Power_data.txt $dir/power_data.gpu.$dtype.$size.csv


	cmd="./Release/cublas_sgemm 1 1 $repeate $dtype $size | tee log.txt"
	echo $cmd; eval $cmd;
	dir=result.tensorcore.$dtype.$size
	mkdir -p $dir
	rm -f $dir/*
	mv log.txt $dir/log.$dtype.$size.txt
	mv Power_data.txt $dir/power_data.tensorcore.$dtype.$size.csv
}

#dtype=2; gemm $dtype

for ((s=1024; s<=1024*32; s *= 2)); do
	for ((t=0; t<=2; t++)); do
		gemm $t $s
    done
done

