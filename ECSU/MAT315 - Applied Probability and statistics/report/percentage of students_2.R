rdata <- read.csv('ReportData.csv')
grde_vector <-  vector(mode = "character",length = length(rdata$StudentGrades))
grde_vector[rdata$StudentGrades == 1] <- "D"
grde_vector[rdata$StudentGrades >1 & rdata$StudentGrades < 5] <- 'C'
grde_vector[rdata$StudentGrades >4 & rdata$StudentGrades < 8] <- 'B'
grde_vector[rdata$StudentGrades >7] <- 'A'
hour_vector <-vector(mode ='character', length = length(rdata$Hours.Worked))
hour_vector[rdata$Hours.Worked == 1] <- 'None'
hour_vector[rdata$Hours.Worked >1 & rdata$Hours.Worked < 4] <- '1-10'
hour_vector[rdata$Hours.Worked >3 & rdata$Hours.Worked < 6] <- '11-20'
hour_vector[rdata$Hours.Worked >5 & rdata$Hours.Worked < 8] <- '21-30'
hour_vector[rdata$Hours.Worked == 8] <- '30+'
# rdata$StudentGrades <- factor(rdata$StudentGrades, levels = c(1,2,3,4,5,6,7,8,9), labels=c('d','c-','c','c+','b-','b','b+','a-','a'))
rdata$Hours.Worked <- factor(rdata$Hours.Worked, levels= c(1,2,3,4,5,6,7,8), labels = c('None','<=5','6-10','11-15','16-20','21-25','26-30','30+'))
grade <- factor(grde_vector, level = c("D","C","B","A"), ordered = TRUE)
rdata <- cbind(rdata, grade)
hour <- factor(hour_vector, level = c('None', '1-10', '11-20', '21-30', '30+'))
rdata <- cbind(rdata, hour)
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

