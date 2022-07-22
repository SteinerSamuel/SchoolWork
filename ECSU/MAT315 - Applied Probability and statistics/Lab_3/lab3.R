DataLab3 <- read.csv("data/lab_3.csv", header= TRUE)
head(DataLab3)
tail(DataLab3)
DataLab3$sex <- factor(DataLab3$sex, levels = c(1,2), labels = c("Male", "Female"))
DataLab3$race <- factor(DataLab3$race, levels = c(1,2,3), labels = c("Black", "White", "Hispanic"))
DataLab3$sex <- factor(DataLab3$sex, levels = c(1,2), labels = c("Male", "Female"))
Table_Sex <- table(DataLab3$sex)
Table_Sex
Table_Sex_Prep <- prop.table(Table_Sex)*100
Table_Sex_Prep
