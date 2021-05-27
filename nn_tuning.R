# Nearest Neighbor Tuning ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)

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


