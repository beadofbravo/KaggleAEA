## libraries
library(tidyverse)
library(vroom)
library(patchwork)
library(DataExplorer)
library(ggmosaic)
library(tidymodels)
library(embed)
library(doParallel)


## cl <- makePSOCKcluster(8)
## registerDoParallel(cl)

empl_access_train <- vroom("./train.csv") %>%
  mutate(ACTION = factor(ACTION))
empl_access_test <- vroom("./test.csv")

my_recipe <- recipe(ACTION ~., data = empl_access_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_factor_predictors(), threshold = .01) %>%
  step_dummy(all_nominal_predictors()) %>%
  prep()













