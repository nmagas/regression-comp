# Mars Model Tuning----

## Load package(s) ----
library(tidyverse)
library(tidymodels)


set.seed(42)

# load required objects ----
load(file = "data/loan_folds.rda")
load(file = "data/loan_recipe.rda")

# Define model ----
mars_model <- mars(num_terms = tune(), prod_degree =tune()) %>% 
  set_mode("regression") %>% 
  set_engine("earth")



# set-up tuning grid ----

# checking parameters
mars_params <- parameters(mars_model)



# define tuning grid
mars_grid <- grid_regular(mars_params, levels = 10)


# workflow ----
mars_workflow <- workflow() %>% 
  add_model(mars_model) %>% 
  add_recipe(loan_recipe)


# Tuning/fitting ----

# Place tuning code in here
mars_tune <- mars_workflow %>% 
  tune_grid(resamples = loan_folds, grid = mars_grid)




# Write out results & workflow

save(mars_tune, mars_workflow, file = "data/mars_tune.rda")
