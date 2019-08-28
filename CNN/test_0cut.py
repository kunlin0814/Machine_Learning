from sys import *
from os import *
import openslide
import numpy
from PIL import Image
import warnings

warnings.simplefilter ('ignore', Image.DecompressionBombWarning)
InputPath="/scratch/kh31516/TCGA/Stomach/TCGA-HE-Image/TCGA_stomach_tumor_image/micro_enrich/"
case='TCGA-BR-4184-01Z-00-DX1.aa0818ba-92ea-4337-b9d3-c891161f5f9a'
OutputPath="/scratch/kh31516/TCGA/Stomach/TCGA-HE-Image/picture_test"
Class='Deplete'
source=openslide.open_slide(InputPath+case+"/"+case+".svs")
makedirs(OutputPath+Class+"/"+case[:-11])
pathSize=int(224)
count=0
for i in range(0,2):
	for j in range(0,2):
                region=numpy.array(source.read_region((i*800,j*600),0, (800,600)))
                for m in range (0,800-pathSize,pathSize):
                        for n in range(0,600-pathSize,pathSize):
                                box = (m, n, (m+224), (n+224))
                                im=Image.fromarray(region)
                                cropped=im.crop (box)
                                r,g,b,a=cropped.split()
                                if (numpy.std(numpy.array(r))+numpy.std(numpy.array(g))+numpy.std(numpy.array(b))>54):
                                        count=count+1
                                        toImage=Image.merge("RGB",(r,g,b))
                                        toImage.save(OutputPath+Class+"/"+case[:-11]+"/"+case[:-11]+"-"+str(count)+".jpg")

#Image.MAX_IMAGE_PIXELS = None