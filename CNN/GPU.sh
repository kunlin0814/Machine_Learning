#PBS -S /bin/bash
#PBS -q gpu_q
#PBS -N testjob
#PBS -l nodes=1:ppn=4:gpus=1
#PBS -l walltime=200:00:00
#PBS -l mem=40gb
#PBS -M kh31516@uga.edu 
#PBS -m ae


cd $PBS_O_WORKDIR
module load Python/2.7.14-foss-2016b
ml CUDA/9.0.176

python run_Model.py