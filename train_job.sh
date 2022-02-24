#!/bin/bash

#$ -M aherman3@nd.edu
#$ -m abe
#$ -q gpu
#$ -l gpu_card=1
#$ -N aherman3_nn_proj

conda activate nn1
python3 CNN.py
