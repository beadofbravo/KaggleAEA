## libraries
library(tidyverse)
library(vroom)
library(patchwork)
library(DataExplorer)
library(ggmosaic)
library(tidymodels)
library(embed)
library(doParallel)


cl <- makePSOCKcluster(8)
registerDoParallel(cl)

empl_access_train <- vroom("./train.csv") %>%
  mutate(ACTION = factor(ACTION))
empl_access_test <- vroom("./test.csv")

my_recipe_crt <- recipe(ACTION ~., data = empl_access_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_factor_predictors(), threshold = .001) %>%
  step_dummy(all_nominal_predictors()) %>%
  prep()


my_mod_crf <- rand_forest(mtry = tune(),
                          min_n = tune(),
                          trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

class_reg_tree_wf <- workflow() %>%
  add_recipe(my_recipe_crt) %>%
  add_model(my_mod_crf)

tuning_grid_crf <- grid_regular(min_n(),
                                mtry(range = c(1, 10)))


folds <- vfold_cv(empl_access_train, v = 5, repeats = 1)

CV_results_crf <- class_reg_tree_wf %>%
  tune_grid(resamples = folds, 
            grid = tuning_grid_crf,
            metrics=metric_set(roc_auc))

bestTune <- CV_results_crf %>%
  select_best("roc_auc")

final_wf_crf <- class_reg_tree_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=empl_access_train)

plr_predictions <- predict(final_wf_crf, new_data = empl_access_test, type = "prob")


amazon_predictions_plr <- plr_predictions %>%
  bind_cols(., empl_access_test) %>%
  select(id, .pred_1) %>%
  rename(ACTION = .pred_1)


vroom_write(x=amazon_predictions_plr, file="./crf.csv", delim=",")

stopCluster(cl)
 





