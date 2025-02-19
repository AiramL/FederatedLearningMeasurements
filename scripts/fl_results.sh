#!/bin/bash

eps=100
numClients=20
numClientsFit=16
bs=128
strategy="random"
#dataset="VeReMi"
#dataset="WiSec"
dataset=CIFAR
model=VGG19

echo "Verifying if the results directory exists"
[ ! -d ../results/classification/raw/$strategy/$dataset/ ] && mkdir -p ../results/classification/raw/$strategy/$dataset/

echo "Starting server"
cd ../src/federated_learning
if [ "$strategy" = "m_fastest" ]
then	
	[[ $(($numClientsFit/2)) = 0 ]] && numClientsFit=1 || numClientsFit=$(($numClientsFit/2))
	python3.12 server.py -ncf=$numClientsFit -nc=$numClients -nor=$eps &
else	
	python3.12 server.py -ncf=$numClientsFit -nc=$numClients -nor=$eps &
fi

echo "Starting clients"
sleep 3

# initialize clients
for i in $(seq $numClients)
do
		python3.12 client.py -md=$model -nc=$numClients -cid=$i -b=$bs -cf=0 -ncf=$numClientsFit >> ../../results/classification/raw/$strategy/$dataset/$model/"client_""$i" &
	echo "Waiting client "$i" initialization"
	sleep 2

done
