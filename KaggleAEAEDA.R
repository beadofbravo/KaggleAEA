## libraries
library(tidyverse)
library(vroom)
library(patchwork)
library(DataExplorer)
library(ggmosaic)
library(tidymodels)
library(embed)

empl_access_train <- vroom("./train.csv") %>%
  mutate(ACTION = factor(ACTION))
empl_access_test <- vroom("./test.csv")
view(empl_access_train)


my_recipe <- recipe(ACTION ~., data = empl_access_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_factor_predictors(), threshold = .01) %>%
  step_dummy(all_nominal_predictors()) %>%
  prep()

prep<- prep(my_recipe)
bake <- bake(prep, new_data = NULL)


#########################
## Logistic Regression ##
#########################

my_mod_lr <- logistic_reg() %>%
  set_engine("glm")

amazon_workflow_lr <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod_lr) %>%
  fit(data = empl_access_train)

amazon_predictions_lr <- predict(amazon_workflow_lr,
                                 new_data = empl_access_test,
                                 type = "prob")

amazon_predictions_lr <- amazon_predictions_lr %>%
  bind_cols(., empl_access_test) %>%
  select(id, .pred_1) %>%
  rename(ACTION = .pred_1)
  
view(amazon_predictions_lr)
vroom_write(x=amazon_predictions_lr, file="./lr.csv", delim=",")



###################################
## Penalized Logistic Regression ##
###################################

my_recipe_plr <- recipe(ACTION ~., data = empl_access_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_factor_predictors(), threshold = .001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  prep()

my_mod_plr <- logistic_reg(mixture = tune(), penalty = tune()) %>%
  set_engine("glmnet")

plr_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod_plr)


tuning_grid_plr <- grid_regular(penalty(),
                                mixture(),
                                levels = 10)

folds <- vfold_cv(empl_access_train, v = 5, repeats = 1)

CV_results <- plr_workflow %>%
  tune_grid(resamples = folds,
            grid = tuning_grid_plr,
            metrics=metric_set(roc_auc))


best_tune_plr <- CV_results %>%
  select_best("roc_auc")


finalize_workflow_plr <- plr_workflow %>%
  finalize_workflow(best_tune_plr) %>%
  fit(data = empl_access_train)

## Predictions

plr_predictions <- predict(finalize_workflow_plr, new_data = empl_access_test, type = "prob")


amazon_predictions_plr <- plr_predictions %>%
  bind_cols(., empl_access_test) %>%
  select(id, .pred_1) %>%
  rename(ACTION = .pred_1)


vroom_write(x=amazon_predictions_plr, file="./plr.csv", delim=",")









