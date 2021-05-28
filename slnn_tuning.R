# Single Layer Neural Network Tuning ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)

set.seed(42)

# load required objects ----
load(file = "data/loan_folds.rda")
load(file = "data/loan_recipe.rda")

# Define model ----
slnn_model <- mlp(hidden_units = tune(), penalty = tune()) %>% 
  set_mode("regression") %>% 
  set_engine("nnet")



# set-up tuning grid ----

# parameters
slnn_params <- parameters(slnn_model)


# define tuning grid

slnn_grid <- grid_regular(slnn_params)


# workflow ----
slnn_workflow <- workflow() %>% 
  add_model(slnn_model) %>% 
  add_recipe(loan_recipe)


# Tuning/fitting ----

slnn_tune <- slnn_workflow %>% 
  tune_grid(resamples = loan_folds, grid = slnn_grid)



# Write out results & workflow

save(slnn_tune, slnn_workflow, file = "data/slnn_tune.rda")













# examining slnn performance

load(file = "data/slnn_tune.rda")

slnn_workflow_tuned <- slnn_workflow %>% 
  finalize_workflow(select_best(slnn_tune, metric = "rmse"))

slnn_results <- fit(slnn_workflow_tuned, loan_train)

metrics <- metric_set(rmse)

predict(slnn_results, new_data = loan_test) %>% 
  bind_cols(loan_test %>% select(money_made_inv)) %>% 
  metrics(truth = money_made_inv, estimate = .pred)

