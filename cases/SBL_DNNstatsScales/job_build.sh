#!/bin/bash
#PBS -N build
### Project code
#PBS -A UCLB0017 
#PBS -l walltime=01:00:00
#PBS -q main
### Merge output and error files
#PBS -j oe
#PBS -k eod
#PBS -l select=1:ncpus=1
### Specify index range of sub-jobs
### Send email on abort, begin and end
#PBS -m abe
### Specify mail recipient
#PBS -M ac5006@columbia.edu

source ~/env_cpptorch
conda activate cpptorch

sh newbuild.sh