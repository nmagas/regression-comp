# Loading Packages----
library(tidyverse)
library(tidymodels)
library(naniar)
library(skimr)
library(corrplot)


# setting seed
set.seed(42)


# reading in data
loan <- read_csv("data/train.csv") %>% 
  mutate_if(is.character, as.factor)



final_loan_test <- read_csv("data/test.csv")%>% 
  mutate_if(is.character, as.factor)






# checking for missingness
miss_var_summary(loan)

skim_without_charts(loan)






loan_corr1 <- loan %>% 
  select(acc_now_delinq, acc_open_past_24mths, annual_inc, avg_cur_bal,
         int_rate, loan_amnt, money_made_inv)




corrplot(cor(loan_corr1),
         method = "number",
         type = "upper")





loan_corr2 <- loan %>% 
  select(mort_acc, num_sats, num_tl_120dpd_2m, out_prncp_inv, tot_cur_bal, money_made_inv)




corrplot(cor(loan_corr2),
         method = "number",
         type = "upper")



  






# splitting data
loan_split <- initial_split(loan, prop = 0.7, strata = money_made_inv)
loan_train <- training(loan_split)
loan_test <- testing(loan_split)




# creating folds
loan_folds <- vfold_cv(loan_train, v = 5, repeats = 3, strata = money_made_inv)




# creating recipe
loan_recipe <- recipe(money_made_inv ~ annual_inc + int_rate + grade + loan_amnt + 
                        out_prncp_inv + term, data = loan_train) %>% 
  step_other(all_nominal(), -all_outcomes(), threshold = 0.15) %>% 
  step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE) %>% 
  step_normalize(all_numeric(), -all_outcomes()) %>% 
  step_zv(all_predictors())







# prepping and baking recipe

loan_recipe %>% 
  prep(loan_train) %>% 
  bake(new_data = NULL)





# saving necessary tuning objects
save(loan_folds, file = "data/loan_folds.rda")
save(loan_recipe, file = "data/loan_recipe.rda")





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





# examining rf performance

load(file = "data/rf_tune.rda")

rf_workflow_tuned <- rf_workflow %>% 
  finalize_workflow(select_best(rf_tune, metric = "rmse"))

rf_results <- fit(rf_workflow_tuned, loan_train)

metrics <- metric_set(rmse)

predict(rf_results, new_data = loan_test) %>% 
  bind_cols(loan_test %>% select(money_made_inv)) %>% 
  metrics(truth = money_made_inv, estimate = .pred)





# rf submission code

rf_final_predictions <- predict(rf_results, new_data = final_loan_test)

submit <- read_csv("data/sampleSubmission.csv") %>% 
  bind_cols(rf_final_predictions) %>% 
  select(-Predicted) %>% 
  rename(Predicted = .pred)

write_csv(file = "reg_results6.csv", submit)
  
  








# examining mars performance

load(file = "data/mars_tune.rda")

mars_workflow_tuned <- mars_workflow %>% 
  finalize_workflow(select_best(mars_tune, metric = "rmse"))

mars_results <- fit(mars_workflow_tuned, loan_train)

metrics <- metric_set(rmse)

predict(mars_results, new_data = loan_test) %>% 
  bind_cols(loan_test %>% select(money_made_inv)) %>% 
  metrics(truth = money_made_inv, estimate = .pred)



# mars submission code
  
mars_final_predictions <- predict(mars_results, new_data = final_loan_test)

submit_mars <- read_csv("data/sampleSubmission.csv") %>% 
  bind_cols(mars_final_predictions) %>% 
  select(-Predicted) %>% 
  rename(Predicted = .pred)

write_csv(file = "reg_results3.csv", submit)























  
  
  














