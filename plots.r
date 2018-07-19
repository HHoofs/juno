require(ggplot2)
require(foreign)

df_ <- read.csv('db_nist.csav', stringsAsFactors = FALSE)
df_$gender <- as.factor(df_$gender)
df_$pattern <- as.factor(df_$pattern)

ggplot(data=df_, aes(df_$pattern, fill=df_$pattern)) + geom_bar(colour='black')
