## libraries
library(tidyverse)
library(vroom)
library(patchwork)
library(DataExplorer)
library(ggmosaic)
library(tidymodels)
library(embed)

empl_access_train <- vroom("./train.csv")
empl_access_test <- vroom("./test.csv")
view(empl_access_train)
ggplot(data = empl_access_train) +
  geom_mosaic(aes(x = MGR_ID, fill = ACTION))


my_recipe <- recipe(ACTION ~., data = empl_access_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_factor_predictors(), threshold = .01) %>%
  step_dummy(all_nominal_predictors()) %>%
  prep()

prep<- prep(my_recipe)
bake <- bake(prep, new_data = NULL)















