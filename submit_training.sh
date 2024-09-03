#!/bin/bash
NNODES=16
mkdir -p /proj/data-eng/blog/torchtitan/outputs
echo bsub -M 512G -J blog -gpu \"num=8/task:mode=exclusive_process\" -n $NNODES -e \"/proj/data-eng/blog/torchtitan/outputs/blog.err\" -o \"/proj/data-eng/blog/torchtitan/outputs/blog.out\" blaunch ./training.sh
bsub -M 512G -J blog -gpu "num=8/task:mode=exclusive_process" -n $NNODES -e "/proj/data-eng/blog/torchtitan/outputs/blog.err" -o "/proj/data-eng/blog/torchtitan/outputs/blog.out" blaunch ./training.sh
