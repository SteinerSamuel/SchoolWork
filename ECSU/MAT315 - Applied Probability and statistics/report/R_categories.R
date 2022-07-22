MTF2016 <- read.csv("z:/MAT315/Lab_3/data/MTFData_2016.csv", header = TRUE)
MTF2016$V2150 <- factor(MTF2016$V2150, levels = c(1,2), labels = c("Male", "Female"))
alc_vector <- vector(mode = "character", length = length(MTF2016$V2104))
head(alc_vector)
alc_vector[MTF2016$V2104 == 0] <- "Never"
alc_vector[MTF2016$V2104 > 0 & MTF2016$V2104 < 5] <-  "1-9"
alc_vector[MTF2016$V2104 == 5] <- "10-19"
alc_vector[MTF2016$V2104 == 6] <- "20-39"
alc_vector[MTF2016$V2104 == 7] <- "40+"
alc <- factor(alc_vector, level = c("Never", "1-9", "10-19", "20-39", "40+"), ordered= TRUE)
MTF2016 <- cbind(MTF2016, alc)
Table_ALC <- table(MTF2016$V2150, MTF2016$alc)
Table_ALC
Table_ALC_Per <- prop.table(Table_ALC, 2)*100
Table_ALC_Per
barplot(Table_ALC_Per, xlab ="Alcholic Drinks during lifetime", col = c("Blue", "Red"), ylab = "Percent", beside = TRUE)
