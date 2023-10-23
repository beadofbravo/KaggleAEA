library(tidyverse)
library(vroom)
library(patchwork)
library(DataExplorer)
library(ggmosaic)
library(tidymodels)
library(embed)
library(doParallel)
library(discrim)
library(naivebayes)

cl <- makePSOCKcluster(8)
registerDoParallel(cl)


empl_access_train <- vroom("./train.csv") %>%
  mutate(ACTION = factor(ACTION))
empl_access_test <- vroom("./test.csv")

my_recipe_nb <- recipe(ACTION ~., data = empl_access_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_factor_predictors(), threshold = .001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), threshold = .91) %>%
  prep()


nb_model <- naive_Bayes(Laplace = tune(), smoothness = tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

nb_workflow <- workflow() %>%
  add_recipe(my_recipe_nb) %>%
  add_model(nb_model)

tuning_grid_nb <- grid_regular(Laplace(),
                               smoothness(),
                               levels = 5)

folds <- vfold_cv(empl_access_train, v = 5, repeats = 1)

nb_results_tune <- nb_workflow %>%
  tune_grid(resamples = folds,
            grid = tuning_grid_nb,
            metrics=metric_set(roc_auc))

bestTune <- nb_results_tune %>%
  select_best("roc_auc")


final_wf_nb <- nb_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=empl_access_train)

nb_predictions <- final_wf_nb %>%
  predict(new_data = empl_access_test, type = "prob")

amazon_predictions_nb <- nb_predictions %>%
  bind_cols(., empl_access_test) %>%
  select(id, .pred_1) %>%
  rename(ACTION = .pred_1)


vroom_write(x=amazon_predictions_nb, file="./nbPCA.csv", delim=",")

stopCluster(cl)

















