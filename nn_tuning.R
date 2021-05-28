# Nearest Neighbor Tuning ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)

set.seed(42)

# load required objects ----
load(file = "data/loan_folds.rda")
load(file = "data/loan_recipe.rda")

# Define model ----
nn_model <- nearest_neighbor(neighbors = tune()) %>% 
  set_mode("regression") %>% 
  set_engine("kknn")


# set-up tuning grid ----

# check parameters
nn_params <- parameters(nn_model)



# define tuning grid
nn_grid <- grid_regular(nn_params, levels = 5)



# workflow ----
nn_workflow <- workflow() %>% 
  add_model(nn_model) %>% 
  add_recipe(loan_recipe)





# Tuning/fitting ----

nn_tune <- nn_workflow %>% 
  tune_grid(resamples = loan_folds, grid = nn_grid)



# Write out results & workflow

save(nn_tune, nn_workflow, file = "data/nn_tune.rda")











# examining nearest neighbors performance


load(file = "data/nn_tune.rda")

nn_workflow_tuned <- nn_workflow %>% 
  finalize_workflow(select_best(nn_tune, metric = "rmse"))

nn_results <- fit(nn_workflow_tuned, loan_train)

metrics <- metric_set(rmse)

predict(nn_results, new_data = loan_test) %>% 
  bind_cols(loan_test %>% select(money_made_inv)) %>% 
  metrics(truth = money_made_inv, estimate = .pred)

