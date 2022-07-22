Name <- c("Eric", "Megan", "Shelby", "Arianna", "Nick")
Status <- c("Happy", "So-so", "Happy", "Happy", "UnHappy")
Height <- c(75, 68, 80, 68, 70)
df <- data.frame(Name, Status, Height)
df
str(df)
df <- data.frame(Name, Status, Height, stringsAsFactors = FALSE)
Table_Status <- table(df$Status)
barplot(Table_Status)
int <- read.csv("Int.csv", header = TRUE)
head(int)
str(int)
int$Sex <- factor(int$Sex, levels = c(1,2), labels = c("Male", "Female"))
head(int$Sex)
Table_Sex <- table(int$Sex)
Table_Sex
int$Int_Rate <-  factor(int$Int_Rate, levels = c(1,2,3), labels = c("Below", "Average", "Above"))
head(int$Int_Rate)
Table_Average <- table(int$Int_Rate)
Table_Average
Table_Sex_Int <- table(int$Sex, int$Int_Rate)
Table_Sex_Int
Table_Sex_Int_Margin <- margin.table(Table_Sex_Int)
Table_Sex_Int_Margin
margin.table(Table_Sex)
margin.table(Table_Average)
Table_Row_Percents <- prop.table(Table_Sex_Int, 1)*100
Table_Row_Percents
barplot(Table_Row_Percents, ylab = "Percents", xlab = "Sex", legend = rownames(Table_Sex_Int), col = c("skyblue", "pink"), beside = TRUE)
Table_Row_Dist <- prop.table(Table_Sex_Int, 2)*100
Table_Row_Dist
barplot(Table_Row_Dist, ylab = "Percents", xlab = "Sex", legend = rownames(Table_Sex_Int), col = c("skyblue", "pink"), beside = TRUE)
