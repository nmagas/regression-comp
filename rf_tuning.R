# Random Forest Tuning ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)


set.seed(42)

# load required objects ----
load(file = "data/loan_folds.rda")
load(file = "data/loan_recipe.rda")

# Define model ----
rf_model <- rand_forest(mtry = tune(), min_n = tune()) %>% 
  set_mode("regression") %>% 
  set_engine("ranger")



# set-up tuning grid ----

# checking parameters
rf_params <- parameters(rf_model) %>% 
  update(mtry = mtry(range = c(1, 10)))



# define tuning grid
rf_grid <- grid_regular(rf_params, levels = 5)


# workflow ----
rf_workflow <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(loan_recipe)


# Tuning/fitting ----

# Place tuning code in here
rf_tune <- rf_workflow %>% 
  tune_grid(resamples = loan_folds, grid = rf_grid)




# Write out results & workflow

save(rf_tune, rf_workflow, file = "data/rf_tune.rda")







# examining rf performance (4 folds, 3 repeats, only 4 predictors)


load(file = "data/rf_tune.rda")

rf_workflow_tuned <- rf_workflow %>% 
  finalize_workflow(select_best(rf_tune, metric = "rmse"))

rf_results <- fit(rf_workflow_tuned, loan_train)

metrics <- metric_set(rmse)

predict(rf_results, new_data = loan_test) %>% 
  bind_cols(loan_test %>% select(money_made_inv)) %>% 
  metrics(truth = money_made_inv, estimate = .pred)




# rf2 submission code

rf_final_predictions <- predict(rf_results, new_data = final_loan_test)

submit_rf <- read_csv("data/sampleSubmission.csv") %>% 
  bind_cols(rf_final_predictions) %>% 
  select(-Predicted) %>% 
  rename(Predicted = .pred)

write_csv(file = "reg_results5.csv", submit_rf)
