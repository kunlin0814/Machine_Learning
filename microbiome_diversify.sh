#PBS -S /bin/bash
#PBS -q batch
#PBS -l nodes=1:ppn=1:AMD
#PBS -l walltime=48:00:00
#PBS -l mem=20gb

cd /scratch/kh31516/TCGA/Stomach_original/Stomach/results/
: "
while read line;
do 
    cd /scratch/kh31516/TCGA/Stomach_original/Stomach/results/$line
    python3 /scratch/kh31516/TCGA/Stomach_original/Stomach/scripts/micro_phylum_enrich.py HumanMicroBiome/${line}.sam-readsID-PhylumFamilySpecies-PhylumSum-sort-fill0 ${line}-TotalReads
    python3 /scratch/kh31516/TCGA/Stomach_original/Stomach/scripts/micro_family_enrich.py HumanMicroBiome/${line}.sam-readsID-PhylumFamilySpecies-FamilySum-sort-fill0 ${line}-TotalReads
    python3 /scratch/kh31516/TCGA/Stomach_original/Stomach/scripts/micro_species_enrich.py HumanMicroBiome/${line}.sam-readsID-PhylumFamilySpecies-SpeciesSum-sort-fill0 ${line}-TotalReads

done < /scratch/kh31516/TCGA/Stomach_original/Stomach/source/TCGA_total_cases.txt

while read line;
do 
    cd /scratch/kh31516/TCGA/Stomach_original/Stomach/results/$line
    cat fuso_enrich_phylum.txt >> /scratch/kh31516/TCGA/Stomach_original/Stomach/results/fuso_phylum_sum.txt
    cat fuso_enrich_family.txt >> /scratch/kh31516/TCGA/Stomach_original/Stomach/results/fuso_family_sum.txt
    cat fuso_enrich_species.txt >> /scratch/kh31516/TCGA/Stomach_original/Stomach/results/fuso_species_sum.txt
done < /scratch/kh31516/TCGA/Stomach_original/Stomach/source/TCGA_total_cases.txt
"

while read line;
do 
    cd /scratch/kh31516/TCGA/Stomach_original/Stomach/results/$line
    python3 /scratch/kh31516/TCGA/Stomach_original/Stomach/scripts/diversity_species.py HumanMicroBiome/${line}.sam-readsID-PhylumFamilySpecies-SpeciesSum-sort ${line}-TotalReads
    python3 /scratch/kh31516/TCGA/Stomach_original/Stomach/scripts/diversity_family.py HumanMicroBiome/${line}.sam-readsID-PhylumFamilySpecies-FamilySum-sort ${line}-TotalReads
    python3 /scratch/kh31516/TCGA/Stomach_original/Stomach/scripts/diversity_phylum.py HumanMicroBiome/${line}.sam-readsID-PhylumFamilySpecies-PhylumSum-sort ${line}-TotalReads
done < /scratch/kh31516/TCGA/Stomach_original/Stomach/source/TCGA_total_cases.txt

while read line;
do 
    cd /scratch/kh31516/TCGA/Stomach_original/Stomach/results/$line
    cat diversity_calculation_phylum.txt >> /scratch/kh31516/TCGA/Stomach_original/Stomach/results/phylum_diversity_sum.txt
    cat diversity_calculation_family.txt >> /scratch/kh31516/TCGA/Stomach_original/Stomach/results/family_diversity_sum.txt
    cat diversity_calculation_species.txt >> /scratch/kh31516/TCGA/Stomach_original/Stomach/results/species_diversity_sum.txt
done < /scratch/kh31516/TCGA/Stomach_original/Stomach/source/TCGA_total_cases.txt