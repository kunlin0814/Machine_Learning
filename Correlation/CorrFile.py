from sys import *
import linecache 
import os  

out1=open(argv[2]+'.blood',"a+")
out2=open(argv[2]+'.adjacent',"a+")
#check whether output file exist
if os.stat(argv[2]+'.blood').st_size == 0:
	out1.write("CaseID"+"\t"+"01"+"\t"+"10"+"\n")
if os.stat(argv[2]+'.adjacent').st_size == 0:
	out2.write("CaseID"+"\t"+"01"+"\t"+"11"+"\n")

P=open(argv[1], "r")
count = len(open(argv[1]).readlines())
#Pvec=P.readlines()
for i in range(count-1):
	sample1=linecache.getline(argv[1],i).split(" ")
	for j in range(i+1,count):
		sample2=linecache.getline(argv[1],j).split(" ")
		if sample1[0][0:12] ==sample2[0][0:12]:
			if len(sample1[0])==len(sample2[0]):
				caseid=sample1[0][0:12]
				if sample1[0][13:15]=='01':
					if sample2[0][13:15]=='10':
						if (sample1[1][-1:].find(' ') != -1) or (sample1[1][-1:].find('\t') != -1) or  (sample1[1][-1:].find('\n') != -1):
							if (sample2[1][-1:].find(' ') != -1) or  (sample2[1][-1:].find('\t') != -1) or  (sample2[1][-1:].find('\n') != -1):
								out1.write(caseid+"\t"+sample1[1][:-1]+"\t"+sample2[1][:-1]+"\n")
							else:
								out1.write(caseid+"\t"+sample1[1][:-1]+"\t"+sample2[1]+"\n")
						else:
							if (sample2[1][-1:].find(' ') != -1) or  (sample2[1][-1:].find('\t') != -1) or  (sample2[1][-1:].find('\n') != -1):
								out1.write(caseid+"\t"+sample1[1]+"\t"+sample2[1][:-1]+"\n")
							else:
								out1.write(caseid+"\t"+sample1[1]+"\t"+sample2[1]+"\n")
					if sample2[0][13:15]=='11':
						if (sample1[1][-1:].find(' ') != -1) or  (sample1[1][-1:].find('\t') != -1) or  (sample1[1][-1:].find('\n') != -1):
							if (sample2[1][-1:].find(' ') != -1) or (sample2[1][-1:].find('\t') != -1) or (sample2[1][-1:].find('\n') != -1):
								out2.write(caseid+"\t"+sample1[1][:-1]+"\t"+sample2[1][:-1]+"\n")
							else:
								out2.write(caseid+"\t"+sample1[1][:-1]+"\t"+sample2[1]+"\n")
						else:
							if (sample2[1][-1:].find(' ') != -1) or  (sample2[1][-1:].find('\t') != -1) or  (sample2[1][-1:].find('\n') != -1):
								out2.write(caseid+"\t"+sample1[1]+"\t"+sample2[1][:-1]+"\n")
							else:
								out2.write(caseid+"\t"+sample1[1]+"\t"+sample2[1]+"\n")
				if sample2[0][13:15]=='01':			
					if sample1[0][13:15]=='10':
						if (sample1[1][-1:].find(' ') != -1) or  (sample1[1][-1:].find('\t') != -1) or (sample1[1][-1:].find('\n') != -1):
							if (sample2[1][-1:].find(' ') != -1) or (sample2[1][-1:].find('\t') != -1) or (sample2[1][-1:].find('\n') != -1):
								out1.write(caseid+"\t"+sample1[1][:-1]+"\t"+sample2[1][:-1]+"\n")
							else:
								out1.write(caseid+"\t"+sample1[1][:-1]+"\t"+sample2[1]+"\n")
						else:
							if (sample2[1][-1:].find(' ') != -1) or (sample2[1][-1:].find('\t') != -1) or (sample2[1][-1:].find('\n') != -1):
								out1.write(caseid+"\t"+sample1[1]+"\t"+sample2[1][:-1]+"\n")
							else:
								out1.write(caseid+"\t"+sample1[1]+"\t"+sample2[1]+"\n")

					if sample1[0][13:15]=='11':
						if (sample1[1][-1:].find(' ') != -1) or (sample1[1][-1:].find('\t') != -1) or (sample1[1][-1:].find('\n') != -1):
							if (sample2[1][-1:].find(' ') != -1) or (sample2[1][-1:].find('\t') != -1) or (sample2[1][-1:].find('\n') != -1):
								out2.write(caseid+"\t"+sample1[1][:-1]+"\t"+sample2[1][:-1]+"\n")
							else:
								out2.write(caseid+"\t"+sample1[1][:-1]+"\t"+sample2[1]+"\n")
						else:
							if (sample2[1][-1:].find(' ') != -1) or (sample2[1][-1:].find('\t') != -1) or (sample2[1][-1:].find('\n') != -1):
								out2.write(caseid+"\t"+sample1[1]+"\t"+sample2[1][:-1]+"\n")
							else:
								out2.write(caseid+"\t"+sample1[1]+"\t"+sample2[1]+"\n")

			else:
				continue
		else:
			break 

out1.close()
out2.close()
