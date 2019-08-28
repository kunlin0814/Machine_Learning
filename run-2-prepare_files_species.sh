#PBS -S /bin/bash
#PBS -q batch
#PBS -l nodes=1:ppn=1:AMD
#PBS -l walltime=48:00:00
#PBS -l mem=20gb
mkdir /scratch/kh31516/TCGA/Stomach_original/Stomach/CORR/Species
cd /scratch/kh31516/TCGA/Stomach_original/Stomach/CORR/
module load Python/2.7.14-foss-2018a
while read line
	do
		for f in */*sam-readsID-PhylumFamilySpecies-SpeciesSum-sort-fill0; 
			do 
				case=$(echo $f |cut -d'/' -f2| cut -d'-' -f1-7)
				count=$(cat $f|grep -i "$line" |cut -d' ' -f2)
				echo "$case" "$count" >> Species/$line
			done
		cat Species/$line|awk '{if(NF!=1)print $0}'|sort > Species/$line-sort
		python /scratch/kh31516/TCGA/Stomach_original/Stomach/scripts/CorrFile.py Species/$line-sort Species/$line
	done < /scratch/kh31516/TCGA/Stomach_original/Stomach/source/speciesList.txt
ls Species/*.blood > Species/blood
ls Species/*.adjacent > Species/adjacent

