rdata <- read.csv('ReportData.csv')
grde_vector <-  vector(mode = "character",length = length(rdata$StudentGrades))
grde_vector[rdata$StudentGrades == 1] <- "D"
grde_vector[rdata$StudentGrades >1 & rdata$StudentGrades < 5] <- 'C'
grde_vector[rdata$StudentGrades >4 & rdata$StudentGrades < 8] <- 'B'
grde_vector[rdata$StudentGrades >7] <- 'A'
# rdata$StudentGrades <- factor(rdata$StudentGrades, levels = c(1,2,3,4,5,6,7,8,9), labels=c('d','c-','c','c+','b-','b','b+','a-','a'))
rdata$Hours.Worked <- factor(rdata$Hours.Worked, levels= c(1,2,3,4,5,6,7,8), labels = c('None','<=5','6-10','11-15','16-20','21-25','26-30','30+'))
grade <- factor(grde_vector, level = c("D","C","B","A"), ordered = TRUE)
rdata <- cbind(rdata, grade)
ttable <-  table(rdata$grade, rdata$Hours.Worked)
ttable
per_ttable <- prop.table(ttable,2)*100
per_ttable
options(digits = 2)
(Xsq <- chisq.test(ttable, correct = FALSE)) 
Xsq
Xsq$expected
workplot <-  barplot(per_ttable, beside = TRUE,
                     main = 'Percentage of Student Grades Within Hours Worked',
                     ylab ='Percent of Students',
                     xlab ='Hours Worked(per Week)',
                     col= c('darkolivegreen4','darkolivegreen3', 'darkolivegreen2', 'darkolivegreen1'))

