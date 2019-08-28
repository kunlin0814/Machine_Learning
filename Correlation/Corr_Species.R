library(minerva)

conn <- file("/scratch/kh31516/TCGA/Stomach_original/Stomach/CORR/Species/blood",open="r")
linn <-readLines(conn)
result<-cbind("Species", "MIC", "spearman","pVspearman","pearson","pVpearson","Case")
write.table(result,append = TRUE,col.names=F,row.names=F,sep="\t",quote=F,file="/scratch/kh31516/TCGA/Stomach_original/Stomach/CORR/Species/blood.result")
for (i in 1:length(linn)){
	Species=unlist(strsplit(linn[i], "[.]"))[1]
	dat <- read.table(paste("/scratch/kh31516/TCGA/Stomach_original/Stomach/CORR/",linn[i],sep=""), header=T)
	x <- dat[,2]
	y <- dat[,3]
	M<-mine(x, y)$MIC
	S<-cor(x,y, method="spearman")
	Sp<-cor.test(x,y,method="spearman")$p.value
	P<-cor(x,y, method="pearson")
	Pp<-cor.test(x,y,method="pearson")$p.value
	count<-max(colSums( dat[,c(2,3)]!= 0))
	result<-cbind(Species,M,S,Sp,P,Pp,count)
	write.table(result,append = TRUE,col.names=F,row.names=F,sep="\t",quote=F,file="/scratch/kh31516/TCGA/Stomach_original/Stomach/CORR/Species/blood.result")
}

close(conn)


conn <- file("/scratch/kh31516/TCGA/Stomach_original/Stomach/CORR/Species/adjacent",open="r")
linn <-readLines(conn)
result<-cbind("Species", "MIC", "spearman","pVspearman","pearson","pVpearson","Case")
write.table(result,append = TRUE,col.names=F,row.names=F,sep="\t",quote=F,file="/scratch/kh31516/TCGA/Stomach_original/Stomach/CORR/Species/adjacent.result")
for (i in 1:length(linn)){
	Species=unlist(strsplit(linn[i], "[.]"))[1]
	dat <- read.table(paste("/scratch/kh31516/TCGA/Stomach_original/Stomach/CORR/",linn[i],sep=""), header=T)
	x <- dat[,2]
	y <- dat[,3]
	M<-mine(x, y)$MIC
	S<-cor(x,y, method="spearman")
	Sp<-cor.test(x,y,method="spearman")$p.value
	P<-cor(x,y, method="pearson")
	Pp<-cor.test(x,y,method="pearson")$p.value
	count<-max(colSums( dat[,c(2,3)]!= 0))
	result<-cbind(Species,M,S,Sp,P,Pp,count)
	write.table(result,append = TRUE,col.names=F,row.names=F,sep="\t",quote=F,file="/scratch/kh31516/TCGA/Stomach_original/Stomach/CORR/Species/adjacent.result")
}

close(conn)

