# Boosted Tree Tuning ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)

set.seed(42)

# load required objects ----
load(file = "data/loan_folds.rda")
load(file = "data/loan_recipe.rda")



# Define model ----

bt_model <- boost_tree(mtry = tune(), min_n = tune(), learn_rate = tune()) %>% 
  set_mode("regression") %>% 
  set_engine("xgboost")





# set-up tuning grid ----

# parameters
bt_params <- parameters(bt_model) %>% 
  update(mtry = mtry(range = c(1, 14)),
         learn_rate = learn_rate(range = c(-5, -0.2)))


# define tuning grid
bt_grid <- grid_regular(bt_params, levels = 5)



# workflow ----

bt_workflow <- workflow() %>% 
  add_model(bt_model) %>% 
  add_recipe(loan_recipe)


# tuning

bt_tune <- bt_workflow %>% 
  tune_grid(resamples = loan_folds, grid = bt_grid)


# Write out results & workflow

save(bt_tune, bt_workflow, file = "data/bt_tune.rda")




