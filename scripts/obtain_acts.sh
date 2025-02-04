#!/bin/bash
# for model_name in 'pythia' 'gemma2' 'llama3'
for model_name in 'gemma2' 'llama3' 'pythia' 
do
    python3 ./geometry/geometry.py --model_name $model_name
done
