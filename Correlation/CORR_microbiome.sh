#PBS -S /bin/bash
#PBS -q batch
#PBS -l nodes=1:ppn=1:AMD
#PBS -N corr_microbiome
#PBS -l walltime=48:00:00
#PBS -l mem=40gb
#PBS -M kh31516@uga.edu 
#PBS -m ae


cd /scratch/kh31516/TCGA/Stomach_original/Stomach/scripts
module load R/3.4.4-foss-2016b-X11-20160819-GACRC
#R CMD BATCH Corr_Phylum.R
#R CMD BATCH Corr_Species.R
R CMD BATCH Corr_Family.R
