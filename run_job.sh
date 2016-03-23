#!/bin/bash
#SBATCH -J PICO_UMLS           # job name
#SBATCH -o PICO_UMLS.o%j       # output and error file name (%j expands to jobID)
#SBATCH -n 16              # total number of mpi tasks requested
#SBATCH -p gpu     # queue (partition) -- normal, development, etc.
#SBATCH -t 02:00:00        # run time (hh:mm:ss) 
#SBATCH --mail-user=byron.wallace@gmail.com
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
module load cuda
module load python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/apps/intel14/hdf5/1.8.12/x86_64/lib/
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python run_UMLS_PICO.py
