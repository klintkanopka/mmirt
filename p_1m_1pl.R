library(tidyverse)
library(RColorBrewer)

setwd('~/projects/mmirt')

ll <- read_csv('output/mix2_1pl_loglik.csv') %>% 
  mutate(iteration = row_number()-1)
beta <- read_csv('output/mix2_1pl_beta.csv')
theta <- read_csv('output/mix2_1pl_theta.csv')

pk <- read_csv('output/person_key.csv')
ik <- read_csv('output/item_key.csv')

ggplot(ll, aes(x = iteration, y = loglikelihood)) +
  geom_line(color='firebrick') + 
  labs(x = 'Iteration', y = 'Average Log Likelihood') +
  theme_bw()

ggplot(beta, aes(x = b01, y = b11)) + 
  geom_point(alpha = 0.2) + 
  labs(x = 'Early Test Difficulty', y = 'Late Test Difficulty') +
  geom_abline(aes(slope=1, intercept=0), color='firebrick', lty=2, alpha=0.5) + 
  theme_bw()

ggplot(theta, aes(x = k, y = theta)) + 
  geom_point(alpha = 0.05) + 
  labs(x = 'k', y = 'Theta') +
  geom_smooth(method='lm', se=F) +
  theme_bw()

summary(lm(theta ~ k, theta))

ggplot(beta, aes(x = b1-b0)) +
  geom_density(color="black", fill="firebrick", alpha=0.9) +
  theme_bw()

mean(beta$b1 > beta$b0)

ggplot(beta) + 
  geom_density(aes(x=b00, fill='b00'), alpha = 0.4) +
  geom_density(aes(x=b01, fill='b01'), alpha = 0.4) +
  geom_density(aes(x=b10, fill='b10'), alpha = 0.4) +
  geom_density(aes(x=b11, fill='b11'), alpha = 0.4) +
  labs(y = 'Density', x = 'Item Parameters') +
  theme_bw()
