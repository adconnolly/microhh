#!/bin/bash
#PBS -N new
### Project code
#PBS -A UCLB0017 
#PBS -l walltime=12:00:00
#PBS -q regular
### Merge output and error files
#PBS -j oe
#PBS -k eod
#PBS -l select=4:ncpus=32:mpiprocs=32
### Specify index range of sub-jobs
### Send email on abort, begin and end
#PBS -m abe
### Specify mail recipient
#PBS -M ac5006@columbia.edu

source ~/env_cpptorch
conda activate cpptorch

sh clear.sh

mpiexec_mpt ./microhh init SBL1800
mpiexec_mpt ./microhh run SBL1800 
