#PBS -S /bin/bash
#PBS -q batch
#PBS -l nodes=1:ppn=1:AMD
#PBS -l walltime=48:00:00
#PBS -l mem=20gb
mkdir /scratch/kh31516/TCGA/Stomach_original/Stomach/CORR/Phylum
cd /scratch/kh31516/TCGA/Stomach_original/Stomach/CORR/
module load Python/2.7.14-foss-2018a
while read line
	do
		for f in */*sam-readsID-PhylumFamilySpecies-PhylumSum-sort-fill0; 
			do 
				case=$(echo $f |cut -d'/' -f2|cut -d'-' -f1-7)
				count=$(cat $f|grep $line|cut -d' ' -f2)
				echo $case $count >> Phylum/$line
			done
		cat Phylum/$line|awk '{if(NF!=1)print $0}'|sort > Phylum/$line-sort
		python /scratch/kh31516/TCGA/Stomach/scripts/CorrFile.py Phylum/$line-sort Phylum/$line
	done< /scratch/kh31516/TCGA/Stomach_original/Stomach/source/phylumList.txt
ls Phylum/*.blood > Phylum/blood
ls Phylum/*.adjacent > Phylum/adjacent