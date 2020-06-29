library(tidyverse)
library(RColorBrewer)

setwd('~/projects/mmirt')

ll <- read_csv('output/loglik.csv')
pk <- read_csv('output/person_key.csv')
ik <- read_csv('output/item_key.csv')
theta <- read_csv('output/theta.csv')
beta <- read_csv('output/beta.csv')

beta$ik <- 0:(nrow(beta) -1)
theta$sid <- 0:(nrow(theta) - 1)

ggplot(ll, aes(x = iteration, y = loglikelihood)) +
  geom_line(color='firebrick') + 
  labs(x = 'Step', y = 'Log Likelihood') +
  theme_bw()




ggplot(theta, aes(x = theta)) +
  geom_density(fill = 'steelblue') +
  theme_bw()

ggplot(theta, aes(x = k)) +
  geom_density(fill = 'steelblue') +
  theme_bw()

ggplot(theta, aes(x = theta, y = k)) +
  geom_point(alpha=0.1) +
  geom_smooth(method = 'lm', se=FALSE) +
  theme_bw()

ggplot(beta, aes(x=b00, y=b10)) +
  geom_point(alpha = 0.2) +
  geom_smooth(method = 'lm', se=FALSE) +
  theme_bw()



ggplot(beta) + 
  geom_density(aes(x=b00, fill='b00'), alpha = 0.5) +
  geom_density(aes(x=b01, fill='b01'), alpha = 0.5) +
  geom_density(aes(x=b10, fill='b10'), alpha = 0.5) +
  geom_density(aes(x=b11, fill='b11'), alpha = 0.5) +
  scale_x_continuous(limits=c(-100, 100)) +
  labs(y = 'Density', x = 'Item Parameters') +
  theme_bw()

d <- read_csv('data/allgrade_Spring_6.csv')

m <- lm(diff ~ sequence_number, data=d)
summary(m)

m2 <- lm(th ~ sequence_number, data=d)
summary(m2)

ths <- d %>% 
  group_by(id) %>% 
  filter(sequence_number == max(sequence_number)) %>%
  ungroup() %>% 
  select(id, th)


theta$sid <- 0:(nrow(theta) - 1)

p <- theta %>% 
  left_join(pk, by='sid') %>% 
  select(-X1) %>% 
  inner_join(ths, by='id') %>% 
  distinct() %>% 
  select(id, sid, th_0, th, theta, k)

ggplot(p, aes(x = th, y = theta, color=k)) +
  geom_point(alpha = 0.01) + 
  labs(x = 'NWEA Theta Estimate', y = 'MMIRT Theta Estimate') +
  scale_color_gradient(low='blue', high='red') +
  theme_bw()

d %>% 
  group_by(itemkey) %>% 
  summarize(n = n()) %>% 
  ggplot(aes(x = n)) +
  geom_histogram() +
  scale_x_continuous(limits=c(0,500))
  theme_bw()
  
ggplot(beta, aes(x=b00-b01, y=b10-b11)) +
  geom_point(alpha = 0.2) +
  geom_smooth(se=FALSE) +
  theme_bw()
