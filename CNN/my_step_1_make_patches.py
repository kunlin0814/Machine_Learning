#!/usr/bin/env python
from sys import *
from os import *
import openslide
import numpy
from PIL import Image
#from openslide.lowlevel import *

#def _load_image(buf, size):
#	MAX_PIXELS_PER_LOAD = (1 << 29) - 1
#	PIXELS_PER_LOAD = 1 << 26

#openslide.lowlevel._load_image = _load_image
InputPath="/scratch/kh31516/TCGA/Stomach/TCGA-HE-Image/TCGA_stomach_tumor_image/micro_enrich/"
case=argv[1]
OutputPath="/scratch/kh31516/TCGA/Stomach/TCGA-HE-Image/picture"
Class=argv[2]
source=openslide.open_slide(InputPath+case+"/"+case+".svs")
makedirs(OutputPath+Class+"/"+case[:-11])
pathSize=int(224)
count=0
for i in range(0,15):
	for j in range(0,15):
		region=numpy.array(source.read_region((i*source.level_dimensions[1][0], j*source.level_dimensions[1][1]),0, source.level_dimensions[1]))
		for m in range (0,source.level_dimensions[1][0]-pathSize,pathSize):
			for n in range(0,source.level_dimensions[1][1]-pathSize,pathSize):
				box = (m, n, m+224, n+224)
				im=Image.fromarray(region)
				cropped=im.crop (box)
				r,g,b,a=cropped.split()
				if (numpy.std(numpy.array(r))+numpy.std(numpy.array(g))+numpy.std(numpy.array(b))>54):
					count=count+1
					#print count, i, j, m, n
					toImage=Image.merge("RGB",(r,g,b))
					toImage.save(OutputPath+Class+"/"+case[:-11]+"/"+case[:-11]+"-"+str(count)+".jpg")

print case[:-11], count, i, j, m, n
#print "the case is "+ str(case)
#print "the region is matrix "+ str(region)
#print "the source level dimension is " + str(source.level_dimensions)
#print " the bix is " + str(box)
#print "the im is "+ str(im)
#print " the cropped is " + str(cropped)

#region=numpy.array(source.read_region((0,0),1,source.level_dimensions[1]))

#x=0
#y=0

#for i in range(0,source.level_dimensions[1][0]-100,100):
#	for j in range(0,source.level_dimensions[1][1]-100,100):
#		box = (j, i, j+100, i+100)
#		im=Image.fromarray(region)
#		cropped=im.crop (box)
#		r,g,b,a=cropped.split()
#		if (numpy.std(numpy.array(r))+numpy.std(numpy.array(g))+numpy.std(numpy.array(b))>54):
#			count=count+1
#			print count
#			toImage=Image.merge("RGB",(r,g,b))
#			toImage.save("/lustre1/jw16567/TCGA-HE-Image/Large/Collective/TCGA-AA-3555-01Z-00-DX1-H/"+str(count)+".jpg")
#print count


