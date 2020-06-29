library(tidyverse)

setwd('~/projects/mmirt/')
beta <- read_csv('output/mix-1-eps-7/1m_1pl_beta.csv')

p <- ggplot(beta, aes(x = b0, y = b1)) + 
  geom_point(alpha = 0.2) + 
  labs(x = 'Early Test Difficulty', y = 'Late Test Difficulty') +
  geom_abline(aes(slope=1, intercept=0), color='cornflowerblue', lty=2) + 
  theme_bw()

ggsave('sdjm.png',
       width=9, height=5, units='in',
       dpi='retina')
