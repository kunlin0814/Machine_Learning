#!/usr/bin/env python
from sys import *
from os import listdir
from random import sample
import math

Input="/scratch/kh31516/TCGA/Stomach/TCGA-HE-Image/picture/"
Output="/scratch/kh31516/TCGA/Stomach/TCGA-HE-Image/4Cluster/model/"
case=argv[1]
Class=argv[2]
percent=float(argv[3])
train=open(Output+Class+"train.txt", "a+")
test=open(Output+Class+"test.txt", "a+")
Picture = [f for f in listdir(Input+Class+"/"+case+"/")]
Picture_tr =sample(Picture, int(math.ceil(len(Picture)*percent)))
for f in Picture_tr:
	print >>train, f
# in python 3, use print(f, file=train)
for f in Picture:
	if f not in Picture_tr:
		print >> test, f

train.close()
test.close()

#D_train=open("D_train.txt", "a+")
#D_test=open("D_test.txt", "a+")
#N = [f for f in listdir("/scratch/jw16567/TCGA-HE-Image/Large/Collective/TCGA-AA-3555-01Z-00-DX1-224/")]
#D_tr =sample(N, int(math.ceil(len(N)*0.8)))
#for f in D_tr:
#	print >> D_train, f

#for f in N:
#	if f not in D_tr:
#		print >> D_test, f

#D_train.close()
#D_test.close()
