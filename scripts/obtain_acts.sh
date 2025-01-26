#!/bin/bash
for model_name in 'pythia' 'gemma2' 'llama3'
do
    python3 ./geometry/obtain.py --model_name $model_name
done
