INBIRTH <- read.csv('INBIRTH.csv')
attach(INBIRTH)
plot(BirthRate ~ Users)
cor(BirthRate, Users)
LinearModel <- lm(BirthRate ~ Users)
LinearModel

plot(LinearModel$residuals ~ Users, ylab = "Residuals",
     xlab = "Users", pch = 16, col='red')
abline(0,0)
plot(LinearModel$residuals ~ LinearModel$fitted.values, ylab = "Residuals", 
     xlab = "Fitted Values", pch = 16, col = "blue")
abline(0,0)
layout(matrix(1:4,2,2))
plot(LinearModel)

logBirthRate <- log(BirthRate)
