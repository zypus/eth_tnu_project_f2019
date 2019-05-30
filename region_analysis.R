library(tidyverse)
library(xgboost)

# Region Analysis

subjects = read_table("data.nosync/relevant_subjects.txt")[[1]]

atlas_name = "harvard_oxford"

region_names = read_table(paste0("data/time_series/", atlas_name, "/region_names.txt"), col_names=FALSE)[[1]] %>% 
  stringr::str_replace_all("[^A-Za-z0-9]+", ".")

dat = tibble()
for (sub in subjects) {
  
  sub_dat = read.csv(paste0("data/time_series/", atlas_name, "/", sub, "_task_stopsignal_bold-", stringr::str_to_upper(atlas_name),"-time_series.txt"), header=FALSE, sep=" ", col.names = region_names) %>% 
    as_tibble() %>% 
      mutate(
        id = sub
      )
  ""
  
  sub_confounds = read_tsv(paste0("data.nosync/derivatives/ds000030/fmriprep/", sub ,"/func/", sub,"_task-stopsignal_bold_confounds-FIX.tsv")) %>% 
    as_tibble()
  
  sub_dat2 = bind_cols(sub_dat, sub_confounds)
  
  sub_events <- read_tsv(paste0("data.nosync/derivatives/ds000030/fmriprep/", sub ,"/func/", sub,"_task-stopsignal_events.tsv")) %>% 
    as_tibble()
  
  sub_dat3 <- sub_dat2 %>% 
  mutate(
    outcome = map_chr(row_number() * 2, function(t) (sub_events[which.min(abs(sub_events$onset-t)),])$TrialOutcome) %>% as_factor()
  )
  
  dat <-  bind_rows(dat, sub_dat3)
}

# Train

train <- sample(c(0,1), NROW(dat), replace = TRUE, prob = c(0.2, 0.8))

encoded_dat <- dat %>% mutate_if(is_character, as_factor)

train_data <- encoded_dat %>% filter(train == 1) %>%  select(one_of(region_names), outcome) # %>% select(-id, -train)
test_data <-  encoded_dat %>% filter(train == 0) %>% select(one_of(region_names), outcome) # select(-id, -train)

imps_per_outcome <- tibble()

for (out in c("SuccessfulGo", "SuccessfulStop", "UnsuccessfulStop", "UnuccessfulGo")) {
  dtrain <- xgb.DMatrix(train_data %>% select(-outcome) %>% as.matrix(), label=train_data$outcome == out)
  dtest <- xgb.DMatrix(test_data %>% select(-outcome) %>% as.matrix(), label=test_data$outcome == out)
  
  watchlist <- list(train=dtrain, evel=dtest)
  
  param <- list()
  
  model <- xgb.train(param, dtrain, nrounds=500, watchlist, early_stopping_rounds = 10)
  
  imp_mat <- xgb.importance(colnames(train_data %>% select(-outcome)), model)
  xgb.plot.importance(imp_mat)
  
  imp_region <- imp_mat %>% as_tibble() %>% 
    filter(Feature %in% region_names) %>% 
    mutate(outcome = out)
  
  imps_per_outcome <- bind_rows(imps_per_outcome, imp_region)
}

imps_per_outcome %>% 
  select(-outcome) %>% 
  group_by(Feature) %>% 
  summarise_all(mean) %>% 
  arrange(-Importance)

imps_per_outcome %>% 
  group_by(outcome) %>% 
  arrange(-Importance) %>% 
  filter(row_number() <= 10) %>% 
  select(Feature, outcome, Importance) %>% 
  group_by(Feature) %>% 
  summarise(count=n(), mean=mean(Importance), std=sd(Importance), min=min(Importance), max=max(Importance), outcomes=paste(outcome, collapse = ", ")) %>% 
  arrange(-count, -max)

imps_per_outcome %>% 
  group_by(outcome) %>% 
  arrange(-Importance) %>% 
  filter(row_number() <= 5) %>% 
  select(Feature, outcome, Importance) %>% 
  arrange(outcome, -Importance)
