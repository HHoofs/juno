require(ggplot2)
require(foreign)

df_ <- read.csv('db_nist.csv', stringsAsFactors = FALSE)
df_$gender <- as.factor(df_$gender)
df_$pattern <- as.factor(df_$pattern)

plot_patterns <-
  ggplot(data=df_, aes(pattern, fill=pattern)) +
  geom_bar(colour='black') +
  scale_x_discrete('Patterns') + scale_y_continuous('Frequency (n)') +
  scale_fill_grey() +
  theme_bw() +
  theme(legend.position = 'none')


ggsave('figs/plot_patterns.png', plot=plot_patterns)
