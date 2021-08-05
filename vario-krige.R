rm(list = ls())
# Load libraries
library(ggplot2)
theme_update(plot.title = element_text(hjust = 0.5))
library(xlsx)
library(sp)
library(gstat)
library(grid)
library(RColorBrewer)
library(viridis)
library(tictoc)
library(beepr)

tic('Total time')

# Import and format data
tic('Data import/format')
note <- 'seen_under10percent_CW_missense-only'
gene <- 'RbcL'
yVar <- 'Kc'
zVar <- 'KcatC'
zName <- zVar
geneLength <- 497
maxd.fraction <- 1
doKrige <- TRUE
filename <- paste('mahaMetric',note,gene,yVar,zVar,sep='_')
setwd('C:/Users/bcalverley/OneDrive - Scripps Research/Documents/0Balch lab/0Covariant metric SCV')
krigedata <- read.delim(paste(filename,".csv",sep=''),sep = ",")
dir.create(file.path(dirname(rstudioapi::getSourceEditorContext()$path),filename))
setwd(file.path(dirname(rstudioapi::getSourceEditorContext()$path),filename))
maxd<-max(dist(krigedata[,1:2]))
maxd
class(krigedata)
coordinates(krigedata)<-c("x","y")
class(krigedata)
toc()

# Calculate empirical variogram and fit variogram model
tic('Variogram')
v<-variogram(z~1, krigedata, cutoff=maxd.fraction*maxd, width=maxd.fraction*maxd/200)
ggplot(v,aes(dist,gamma)) + geom_point(size=.5)
vmf <- fit.variogram(v, vgm(c('Nug','Wav','Exp','Sph','Gau','Exc','Mat','Ste','Cir','Lin','Pen','Per','Hol','Log','Bes','Pow','Spl')),debug.level = 2)
vLine <- variogramLine(vmf, maxdist = maxd.fraction*maxd,n=1000)
vPlot <- ggplot(v,aes(dist,gamma)) + geom_point(size=.5) + geom_line(data = vLine,colour='blue') + ggtitle(paste(gene,yVar,zName,'variogram',sep=' '))
vPlot
if (doKrige) {
# ggsave(paste(filename,'variogram.png',sep='_'),plot=vPlot)
toc()

# Krige cross validation and test accuracy of predictions
tic('Kriging cross-validation')
nMin <- 5
nMax <- 50
kcv <- krige.cv(z~1, locations = krigedata, model=vmf, nmin=nMin, nmax=nMax)
pearson <- cor.test(kcv$var1.pred, kcv$observed, method=c("pearson"))
print(pearson)
setwd(dirname(rstudioapi::getSourceEditorContext()$path))
write.table(data.frame(filename,pearson$estimate,maxd.fraction,format(Sys.time(), '%Y%m%d%H%M%S')), file = "pearson_R_values.csv", sep = ",", append = TRUE, quote = FALSE, col.names = FALSE, row.names = FALSE)
setwd(file.path(dirname(rstudioapi::getSourceEditorContext()$path),filename))
toc()

# Residual plots
tic('Residuals')
resPlot <- ggplot(as(kcv,'data.frame'),aes(var1.pred,residual)) + geom_point(size=.1) + geom_hline(yintercept=0,linetype=2) + ggtitle(paste(gene,zName,'OK prediction residuals',sep=' '))
resPlot
ggsave(paste(filename,'predVresidual.png',sep='_'),plot=resPlot)
ggplot(as(kcv,'data.frame'),aes(observed,var1.pred)) + geom_point() + geom_abline(slope=1,intercept=0)
toc()

# Perform kriging
tic('Kriging')
krigegrid<-expand.grid(x=seq(min(krigedata$x),max(krigedata$x),by=(max(krigedata$x)-min(krigedata$x))/100), y=seq(min(krigedata$y),max(krigedata$y),by=(max(krigedata$y)-min(krigedata$y))/100))
coordinates(krigegrid)=~x+y
gridded(krigegrid)=TRUE
krigemap <- krige(z~1, locations=krigedata, newdata=krigegrid, model = vmf,nmin=nMin, nmax=nMax)
toc()

# Create kriging landscapes
tic('Landscapes')
krigeDF <- as(krigemap,'data.frame')
krigepdf <- as(krigedata,'data.frame')
krigeplot <- try(ggplot(krigeDF,aes(x,y)) + geom_tile(aes(fill = var1.pred)) + geom_point(data=krigepdf[,1:2],size=.1,color='white')+ coord_equal() + scale_fill_viridis(discrete = FALSE) + ggtitle(paste(gene,zName,'OK prediction r =',signif(pearson$estimate,3),sep=' ')) + xlab('Sequence position') + ylab(yVar) + geom_contour(aes(z=var1.var,colour=..level..),size=.2,breaks = quantile(krigeDF$var1.var, c(0,.05,.1,0.25, 0.5, 0.75))) + labs(fill = zName,colour='Variance') + scale_colour_gradient(low='black',high='white'))# + theme(aspect.ratio=1))
if(class(krigeplot) %in% 'try-error') {next}
ggsave(paste(filename,'OKplot.png',sep='_'),plot=krigeplot)
varPlot <- ggplot(krigeDF,aes(x,y)) + geom_tile(aes(fill = var1.var)) + geom_point(data=krigepdf[,1:2],size=.1) + coord_equal() + scale_fill_distiller(palette = 'Reds',direction=1) + ggtitle(paste(gene,'LDL:HDL OK variance r =',pearson$estimate,sep=' ')) + xlab('Sequence position') + ylab(yVar) + labs(fill = "Variance")# + theme(aspect.ratio=1)
ggsave(paste(filename,'Varplot.png',sep='_'),plot=varPlot)
toc()
}

toc()
toc()
beep(2)