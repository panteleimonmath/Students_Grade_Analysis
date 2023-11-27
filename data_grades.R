### Data Analysis of Students' Grades ###

library(MASS)

dfmath <- read.csv("C:\\Users\\mathe\\Documents\\Students_Grade_Analysis\\student-mat.csv", sep = ";", header = TRUE)
dfpor <- read.csv("C:\\Users\\mathe\\Documents\\Students_Grade_Analysis\\student-por.csv", sep = ";", header = TRUE)



df <- merge(dfmath,dfpor,by=c("school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"))
colnames(df)
g3x <- df$G3.x
g3y <- df$G3.y

df <- df[,!colnames(df) %in% c('G1.x','G2.x','G1.y','G2.y','G3.y','G3.x')]
cols <- names(sapply(df, class) == 'factor')


for (i in 1:length(cols)){
  col_num <- cols[i]
  df[,col_num] <- as.numeric(df[,col_num])
}


data.pca <- prcomp(df, tol = 0.05)
dfpca <- predict(data.pca, newdata = df)


modelx <- lm(g3x ~ ., data = data.frame(dfpca))
modely <- lm(g3y ~ ., data = data.frame(dfpca))
summary(modelx)
summary(modely)

step.modelx <- stepAIC(modelx, direction = "both", trace = FALSE)
step.modely <- stepAIC(modely, direction = "both", trace = FALSE)
summary(step.modely)
