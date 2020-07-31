library(tidyverse)
library(RColorBrewer)
library(stargazer)
conflict_prefer("filter", "dplyr")

setwd('~/projects/mmirt/')

ll <- read_csv('output/mix-1-eps-7/1m_1pl_loglik.csv') %>% 
  mutate(iteration = row_number()-1)
beta <- read_csv('output/mix-1-eps-7/1m_1pl_beta.csv')
theta <- read_csv('output/mix-1-eps-7/1m_1pl_theta.csv')

pk <- read_csv('output/mix-1-eps-7/person_key.csv') %>% 
  select(-X1) %>% 
  distinct()

ik <- read_csv('output/mix-1-eps-7/item_key.csv') %>% 
  select(-X1)

full <- read_csv('data/allgrade_Spring_6.csv')

standardize <- function(x){
  (x - mean(x, na.rm=TRUE)) / sd(x, na.rm=TRUE)
}

# add linking indices
theta$sid <- 0:(nrow(theta)-1)
beta$ik <- 0:(nrow(beta)-1)

# build item data
i <- full %>% 
  group_by(itemkey) %>% 
  summarize(diff = mean(diff),
            p = mean(resp),
            n = n(),
            mean_lrt = mean(lrt),
            sd_lrt = sd(lrt),
            mean_seq = mean(sequence_number),
            sd_seq = sd(sequence_number),
            .groups='drop')

i <- ik %>% 
  inner_join(i, by='itemkey') %>% 
  inner_join(beta, by='ik')


# build person data
p <- full %>% 
  group_by(id) %>% 
  filter(sequence_number == max(sequence_number)) %>% 
  ungroup() %>% 
  select(id, nwea_theta = th) %>% 
  distinct()
  
p_2 <- full %>% 
  distinct() %>% 
  group_by(itemkey) %>% 
  mutate(lrt = standardize(lrt)) %>% 
  ungroup() %>% 
  group_by(id) %>% 
  summarize(p_correct = mean(resp),
            mean_lrt = mean(lrt),
            sd_lrt = sd(lrt),
            n_items = n(),
            .groups='drop')

p <- p %>% 
  inner_join(p_2, by='id') %>% 
  inner_join(pk, by='id') %>% 
  inner_join(theta, by='sid')

# person plots

m <- lm(theta~k, data=p)

ggplot(p, aes(x=k, y=theta)) + 
  geom_hex(bins=50) +
  geom_smooth(method=lm, formula=y~x, color='firebrick', se=F) +
  theme_bw()


sigmoid <- function(z) 1 / (1 + exp(-z))
mix_param <- function(s) sigmoid( (20-s)/10)

th <- seq(from=-2, to=2, by=0.01)

b0 <- 0
b1 <- median(i$b1[i$n>500] - i$b0[i$n>500])

irf_0 <- sigmoid(th - b0)
irf_1 <- sigmoid(th - b1)

d <- data.frame(theta = th,
                item_1 = mix_param(1) * irf_0 + (1 - mix_param(1)) * irf_1,
                item_20 = mix_param(20) * irf_0 + (1 - mix_param(20)) * irf_1,
                item_40 = mix_param(40) * irf_0 + (1 - mix_param(40)) * irf_1) %>% 
  pivot_longer(-theta, names_to='Item Position', values_to = 'p')

ggplot(d, aes(x = theta, y=p, color=`Item Position`)) +
  geom_line()
