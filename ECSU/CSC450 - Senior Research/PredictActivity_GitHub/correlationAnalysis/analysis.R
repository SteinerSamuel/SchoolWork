# This R file contains the the correlation analysis of the Data collected from the
# GitHub API. This is to make sure we arent looking at features which are too corellated
# correlation set at 0.7 
library(readr)
library(Hmisc)
library(dendextend)
rm(list = ls()) ## this will delete all objects in the workspace

# Loads in the data
result <- read_csv("result.csv")

#gets the labels on their own
repository_label <- result$Repository
status_label <- result$Status

# removes the labels from the dataframe
result <-  result[,3:106]
res.matrix <- data.matrix(result)

# Does a correlation analysis based on the spearman rho function
rc <- rcorr(as.matrix(result), type="spearman")

# creates a heatmap of the correlation
col<- colorRampPalette(c("blue", "white", "red"))(200)
heatmap(x = rc$r, col = col, symm = TRUE)

# makes a dendrograh of the spearman rho function
v <- varclus(data.matrix(result),similarity = "spearman",trans = "square")
plot(v, cex = .6)

# grabs the dendrogrpah and returns a list of the the non removed features 
v.dendro <- as.dendrogram(v$hclust)

# height here is 1- pearson rho since we want to remove all values of .7 or higher we cut it at .3
v.d.cut <- cut(v.dendro, h=.3)
  
max <- as.integer(nleaves(v.d.cut$upper))
label.list <- list()
for (i in 1:max){
  label.list[[i]] <- as.character(labels(v.d.cut$lower[[i]])[1])
  print(labels(v.d.cut$lower[[i]])[1])
}
# creates a text file with the column names of the non removed features
lapply(label.list, write, "non_removed.txt", append=TRUE, ncolumns=1000)

# Creates the final dendrograph of the non removed  features
dendro <- v.d.cut$upper %>% set("labels",label.list)

plot(dendro, cex= .2, main= "Correlative Analysis After Cut", ylab = "1 - Spearman r^2")

resultmod <- result[,-grep("24_21" , names(result))]
rc2 <- rcorr(as.matrix(resultmod), type="spearman")

heatmap(x = rc2$r, col = col, symm = TRUE)

v2 <- varclus(data.matrix(resultmod),similarity = "spearman",trans = "square")
plot(v2, cex = .6)

v.dendro <- as.dendrogram(v2$hclust)
v.d.cut <- cut(v.dendro, h=.3)
max <- as.integer(nleaves(v.d.cut$upper))
label.list2 <- list()
for (i in 1:max){
  label.list2[[i]] <- as.character(labels(v.d.cut$lower[[i]])[1])
  print(labels(v.d.cut$lower[[i]])[1])
}

lapply(label.list2, write, "non_removed2.txt", append=TRUE, ncolumns=1000)
dendro2 <- v.d.cut$upper %>% set("labels",label.list2)
plot(dendro2, cex= .2, main= "Correlative Analysis After Cut", ylab = "1 - Spearman r^2")
