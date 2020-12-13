#!/bin/sh
#SBATCH --job-name=serial_job_test    # Job name
#SBATCH --mail-type=ALL               # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=shantanughosh@ufl.edu   # Where to send mail
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --mem=16gb                   # Memory limit
#SBATCH --time=20:05:00               # Time limit hrs:min:sec

#SBATCH --output=GAN_log_%j.out   # Standard output and error log

#SBATCH --account=cis6930
#SBATCH --qos=cis6930
#SBATCH --partition=gpu
#SBATCH --gpus=1

pwd; hostname; date

module load python

echo "GAN"

python3 -u GAN_Manager.py > GAN_1000.out

date
