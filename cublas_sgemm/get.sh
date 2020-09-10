DIR=/home/pengdadi/projects/sgemm/cublas_sgemm
NAME=res.nvvp

#ssh -t localhost -p 10023 "cd ${DIR} && ./prof.sh"
ssh -t rwbc-v100.m.gsic.titech.ac.jp "cd ${DIR} && ./prof.sh"
if [ -e $NAME ]; then
	rm -rf res.nvvp
fi
scp rwbc-v100.m.gsic.titech.ac.jp:${DIR}/$NAME .
