#PBS -S /bin/bash
#PBS -q batch
#PBS -l nodes=1:ppn=1:AMD
#PBS -l walltime=48:00:00
#PBS -l mem=20gb
mkdir /scratch/kh31516/TCGA/Stomach_original/Stomach/CORR/Family
cd /scratch/kh31516/TCGA/Stomach_original/Stomach/CORR/
module load Python/2.7.14-foss-2018a
while read line
	do
		for f in */*sam-readsID-PhylumFamilySpecies-FamilySum-sort-fill0; 
			do 
				case=$(echo $f |cut -d'/' -f2|cut -d'-' -f1-7)
				count=$(cat $f|grep $line|cut -d' ' -f2)
				echo $case $count >> Family/$line
			done
		cat Family/$line|awk '{if(NF!=1)print $0}'|sort >Family/$line-sort
		python /scratch/kh31516/TCGA/Stomach/scripts/CorrFile.py Family/$line-sort Family/$line
	done < /scratch/kh31516/TCGA/Stomach_original/Stomach/source/familyList.txt
ls Family/*.blood > Family/blood
ls Family/*.adjacent > Family/adjacent