#PBS -S /bin/bash 
#PBS -q batch
#PBS -l nodes=1:ppn=1:AMD
#PBS -l walltime=500:00:00
#PBS -l pmem=200gb
#PBS -N svs2jpgDepleted
#PBS -M kh31516@uga.edu
#PBS -m ae

module load OpenSlide/3.4.1-foss-2016b 
module load Python/2.7.14-foss-2016b
module load OpenSlide-Python/1.1.1-foss-2016b-Python-2.7.14
cd /scratch/kh31516/TCGA/Stomach/TCGA-HE-Image/TCGA_stomach_tumor_image/micro_deplete/micro_deplete_1
while read line; 
    
    do python my_step_1_make_patches.py $line Depleted; 
    
    done < /scratch/kh31516/TCGA/Stomach/source/Depleted-file1

