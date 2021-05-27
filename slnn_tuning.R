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
