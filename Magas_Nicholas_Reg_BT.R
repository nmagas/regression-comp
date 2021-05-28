# Model Setup, Tuning, and Submission for Boosted Tree model all included in this script
# GitHub link is https://github.com/nmagas/regression-comp.git


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





# Boosted Tree Tuning----


# loading necessary tuning objects
load(file = "data/loan_folds.rda")
load(file = "data/loan_recipe.rda")



# defining model

bt_model <- boost_tree(mtry = tune(), min_n = tune(), learn_rate = tune()) %>% 
  set_mode("regression") %>% 
  set_engine("xgboost")




# parameters
bt_params <- parameters(bt_model) %>% 
  update(mtry = mtry(range = c(1, 14)),
         learn_rate = learn_rate(range = c(-5, -0.2)))


# defining tuning grid
bt_grid <- grid_regular(bt_params, levels = 5)



# creating workflow

bt_workflow <- workflow() %>% 
  add_model(bt_model) %>% 
  add_recipe(loan_recipe)




# tuning
bt_tune <- bt_workflow %>% 
  tune_grid(resamples = loan_folds, grid = bt_grid)





# Writing out results & workflow
save(bt_tune, bt_workflow, file = "data/bt_tune.rda")







# examining bt performance

load(file = "data/bt_tune.rda")

bt_workflow_tuned <- bt_workflow %>% 
  finalize_workflow(select_best(bt_tune, metric = "rmse"))

bt_results <- fit(bt_workflow_tuned, loan_train)

metrics <- metric_set(rmse)

predict(bt_results, new_data = loan_test) %>% 
  bind_cols(loan_test %>% select(money_made_inv)) %>% 
  metrics(truth = money_made_inv, estimate = .pred)




# bt submission code

bt_final_predictions <- predict(bt_results, new_data = final_loan_test)

bt_submit <- read_csv("data/sampleSubmission.csv") %>% 
  bind_cols(bt_final_predictions) %>% 
  select(-Predicted) %>% 
  rename(Predicted = .pred)

write_csv(file = "reg_results8.csv", bt_submit)
