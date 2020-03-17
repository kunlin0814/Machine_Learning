#PBS -S /bin/bash
#PBS -q gpu_q
#PBS -N test_model_Enrcih
#PBS -l nodes=1:ppn=4:gpus=1:K40:default
#PBS -l walltime=200:00:00
#PBS -l mem=60gb
#PBS -M kh31516@uga.edu 
#PBS -m ae


cd /scratch/kh31516/TCGA/Stomach/TCGA-HE-Image/4Cluster/model
ml CUDA/9.0.176
ml Keras/2.2.2-foss-2018a-Python-3.6.4
ml matplotlib/2.1.2-foss-2018a-Python-3.6.4-tf-1.10.1


while read line ;
do 
python run_Model_35.py $line
done < /scratch/kh31516/TCGA/Stomach/TCGA-HE-Image/source/4Cluster_cases/enrich_case.txt
