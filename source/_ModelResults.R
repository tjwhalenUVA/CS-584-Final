
cmGraph <- function(df){
  cm <-
    df %>%
    group_by(true, pred) %>%
    summarise(N = n()) %>%
    ungroup() %>%
    mutate(color=if_else(true == pred, 'green', 'red'), 
           true = factor(true, 
                         levels = c("male", "female"), 
                         order = T)) %>%
    ggplot() +
    geom_point(data=df %>%
                 mutate(color=if_else(true == pred, 'green', 'red'), 
                        true = factor(true, 
                                      levels = c("male", "female"), 
                                      order = T)), 
               aes(x=true, 
                   y=pred, 
                   color=color), 
               position='jitter', 
               alpha=0.5) +
    geom_text(aes(x=true, 
                  y=pred, 
                  label=N), 
              size=10, 
              show.legend = F) +
    theme_classic() +
    scale_color_manual(values = c('green', 'red'))+
    scale_x_discrete(position = "top") +
    labs(x = 'Actual', 
         y='Predicted')
}

modelResults <- function(model){
  f.f <- table(model$result)[1]
  f.m <- table(model$result)[2]
  m.f <- table(model$result)[3]
  m.m <- table(model$result)[4]
  tot <- sum(table(model$result))
  
  acc <- (f.f + m.m) / tot
  male_acc <- m.m / (m.m + m.f)
  female_acc <- f.f / (f.f + f.m)
  
  
  df <- NULL
  df$Model <- c(model$Name)
  df$Accuracy <- c(acc)
  df$MaleAccuracy <- c(male_acc)
  df$FemaleAccuracy <- c(female_acc)
  
  return(data.frame(df))
}

resultsDF <- 
  bind_rows(modelResults(knn),
            modelResults(knn.pca),
            modelResults(dt),
            modelResults(rf),
            modelResults(svm),
            modelResults(svm.pca),
            modelResults(lr),
            modelResults(lr.norm))

resultsDT <- function(df, model){
  dt <-
    resultsDF %>%
    datatable(., 
              rownames=F, 
              options = list(paging=F, 
                             dom = 't')) %>%
    formatPercentage(
      c('Accuracy',
        'MaleAccuracy',
        'FemaleAccuracy'),
      digits=2) %>% 
    formatStyle(
      'Model',
      target = 'row',
      backgroundColor = styleEqual(model, 'green'), 
      color = styleEqual(model, 'white'), 
      fontWeight = styleEqual(model, 'bold')
    )
  return(dt)
}

model_accuracy <- function(model){
  return(resultsDF[resultsDF$Model == model$Name,]$Accuracy)
}


best_params <- function(model){
  finalStr <- ''
  for(p in model$params$Parameter){
    paramStr <-
      paste(p, 
            model$params[model$params$Parameter == p, ]$Value, 
            sep=' = ')
    if(finalStr == ''){
      finalStr <- paramStr
    } else{
      finalStr <- paste(finalStr, 
                        paramStr, 
                        sep = ', ') 
    }
  }
  return(finalStr)
}



#KNN====
knn_cm <- cmGraph(knn$result)
knn_result_dt <- resultsDT(df = resultsDF, model = 'K-Nearest Neighbors')


knn_best.model <-
  knn$GSresult %>%
  filter(param_algorithm == knn$params[knn$params$Parameter == "algorithm", ]$Value)

knn_score.graph <- 
  knn_best.model %>%
  select(param_n_neighbors, param_weights, param_p, 
         mean_test_score, mean_train_score) %>%
  mutate(Parameters = paste(param_weights, param_p, sep="_")) %>%
  ggplot(aes(x=param_n_neighbors, color=Parameters, group=Parameters)) +
  geom_point(aes(y=mean_test_score)) +
  geom_line(aes(y=mean_test_score)) +
  scale_y_continuous(labels = scales::percent) +
  scale_x_continuous(breaks = knn_best.model$param_n_neighbors) +
  labs(title='Cross Validation Results', 
       x='K Neighbors', 
       y='CV Test Score') +
  theme_classic()

#KNN PCA====
knn.pca_cm <- cmGraph(knn.pca$result)
knn.pca_result_dt <- resultsDT(df = resultsDF, model = 'K-Nearest Neighbors (PCA)')

knn.pca_best.model <-
  knn.pca$GSresult %>%
  filter(param_algorithm == knn.pca$params[knn.pca$params$Parameter == "algorithm", ]$Value)

knn.pca_score.graph <- 
  knn.pca_best.model %>%
  select(param_n_neighbors, param_weights, param_p, 
         mean_test_score, mean_train_score) %>%
  mutate(Parameters = paste(param_weights, param_p, sep="_")) %>%
  ggplot(aes(x=param_n_neighbors, color=Parameters, group=Parameters)) +
  geom_point(aes(y=mean_test_score)) +
  geom_line(aes(y=mean_test_score)) +
  scale_y_continuous(labels = scales::percent) +
  scale_x_continuous(breaks = knn.pca_best.model$param_n_neighbors) +
  labs(title='Cross Validation Results', 
       x='K Neighbors', 
       y='CV Test Score') +
  theme_classic()



#DT====
dt_cm <- cmGraph(dt$result)
dt_result_dt <- resultsDT(df = resultsDF, model = 'Decision Tree')

dt_feature.graph <- 
  dt$tree %>%
  select(feature_name, feature_importance) %>%
  unique(.) %>%
  ggplot(aes(x = reorder(feature_name, -feature_importance))) +
  geom_bar(aes(y=feature_importance), 
           stat = 'identity', 
           fill='blue') +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 45, 
                                   hjust = 1)) +
  labs(title='Feature Importance', 
       x='Feature', 
       y='Importance')



#RF====
rf_cm <- cmGraph(rf$result)
rf_result_dt <- resultsDT(df = resultsDF, model = 'Random Forest')

rf_result.graph <-
  rf$CVresult %>%
  select(param_criterion, 
         param_max_depth, 
         param_n_estimators, 
         Mean = mean_test_score) %>%
  mutate(extra_param = paste(param_criterion, 
                             param_max_depth, 
                             sep=', ')) %>%
  select(-param_criterion, 
         -param_max_depth) %>%
  gather(Split, Score, -param_n_estimators, -extra_param) %>%
  mutate(Split = paste(Split, extra_param, sep=' - ')) %>%
  select(-extra_param) %>%
  ggplot(aes(x=param_n_estimators, y=Score, color=Split)) +
  geom_line(show.legend = F) +
  theme_classic() +
  scale_y_continuous(labels = scales::percent) +
  labs(title='Cross Validation Test Scores', 
       x='Number of Estimators', 
       y='Test Score')

#SVM====
svm_cm <- cmGraph(svm$result)
svm_result_dt <- resultsDT(df = resultsDF, model = 'Support Vector Machine')

svm_result.graph <-
  svm$CVresult %>%
  select(param_C, 
         Mean = mean_test_score, 
         Split_0 = split0_test_score, 
         Split_1 = split1_test_score, 
         Split_2 = split2_test_score, 
         Split_3 = split3_test_score, 
         Split_4 = split4_test_score) %>%
  gather(Split, Score, -param_C) %>%
  ggplot(aes(x=param_C, y=Score, color=Split)) +
  geom_line() +
  theme_classic() +
  scale_y_continuous(labels = scales::percent) +
  labs(title='Cross Validation Test Scores', 
       x='Penalty Parameter (C)', 
       y='Test Score')

#SVM PCA====
svm.pca_cm <- cmGraph(svm.pca$result)
svm.pca_result_dt <- resultsDT(df = resultsDF, model = 'Support Vector Machine (PCA)')

svm.pca_result.graph <-
  svm.pca$CVresult %>%
  select(param_C, 
         Mean = mean_test_score, 
         Split_0 = split0_test_score, 
         Split_1 = split1_test_score, 
         Split_2 = split2_test_score, 
         Split_3 = split3_test_score, 
         Split_4 = split4_test_score) %>%
  gather(Split, Score, -param_C) %>%
  ggplot(aes(x=param_C, y=Score, color=Split)) +
  geom_line() +
  theme_classic() +
  scale_y_continuous(labels = scales::percent) +
  labs(title='Cross Validation Test Scores', 
       x='Penalty Parameter (C)', 
       y='Test Score')

#LR====
lr_cm <- cmGraph(lr$result)
lr_result_dt <- resultsDT(df = resultsDF, model = 'Logistic Regression')


lr$CVresult %>%
  names()

lr_result.graph <-
  lr$CVresult %>%
  select(param_C, 
         param_fit_intercept, 
         param_penalty, 
         Mean = mean_test_score) %>%
  mutate(extra_param = paste(param_fit_intercept, 
                             param_penalty, 
                             sep=', ')) %>%
  select(-param_fit_intercept, 
         -param_penalty) %>%
  gather(Split, Score, -param_C, -extra_param) %>%
  mutate(Split = paste(Split, extra_param, sep=' - ')) %>%
  select(-extra_param) %>%
  ggplot(aes(x=param_C, y=Score, color=Split)) +
  geom_line() +
  theme_classic() +
  scale_y_continuous(labels = scales::percent) +
  labs(title='Cross Validation Test Scores', 
       x='C', 
       y='Test Score')

#LR (PCA)====
lr.norm_cm <- cmGraph(lr.norm$result)
lr.norm_result_dt <- resultsDT(df = resultsDF, model = 'Logistic Regression (Normal)')

lr.norm_result.graph <-
  lr.norm$CVresult %>%
  select(param_C, 
         param_fit_intercept, 
         param_penalty, 
         Mean = mean_test_score) %>%
  mutate(extra_param = paste(param_fit_intercept, 
                             param_penalty, 
                             sep=', ')) %>%
  select(-param_fit_intercept, 
         -param_penalty) %>%
  gather(Split, Score, -param_C, -extra_param) %>%
  mutate(Split = paste(Split, extra_param, sep=' - ')) %>%
  select(-extra_param) %>%
  ggplot(aes(x=param_C, y=Score, color=Split)) +
  geom_line() +
  theme_classic() +
  scale_y_continuous(labels = scales::percent) +
  labs(title='Cross Validation Test Scores', 
       x='C', 
       y='Test Score')





#ROC Curves
model_roc <-
  bind_rows(
    mutate(knn$roc, Model = 'KNN'), 
    mutate(knn.pca$roc, Model = 'KNN (PCA)'),
    mutate(dt$roc, Model = 'DT'), 
    mutate(rf$roc, Model = 'RF'), 
    mutate(svm$roc, Model = 'SVM'), 
    mutate(svm.pca$roc, Model = 'SVM (PCA)'), 
    mutate(lr$roc, Model = 'LogR'), 
    mutate(lr.norm$roc, Model = 'LogR (Normal)')
  ) %>%
  ggplot(aes(x=fpr, y=tpr, color=Model)) +
  geom_line() +
  geom_segment(aes(x=0, y=0, 
                   xend=1, yend=1), 
               color='black', 
               show.legend = F) +
  geom_point(aes(x=0, y=1), 
             color='black', 
             show.legend = F) +
  theme_classic() +
  labs(title='ROC', 
       x='False Positive Rate', 
       y='True Positive Rate')


#AUC Table====
AUC <- NULL

AUC$Model <-
  c(knn$Name, 
    knn.pca$Name, 
    dt$Name, 
    rf$Name, 
    svm$Name, 
    svm.pca$Name, 
    lr$Name,
    lr.norm$Name)

AUC$AUC <-
  c(knn$auc$auc, 
    knn.pca$auc$auc, 
    dt$auc$auc, 
    rf$auc$auc, 
    svm$auc$auc, 
    svm.pca$auc$auc, 
    lr$auc$auc,
    lr.norm$auc$auc)

AUC <- data.frame(AUC)



maxAcc <- max(resultsDF$Accuracy)
minAcc <- min(resultsDF$Accuracy)

model_criteria <-
  resultsDF %>%
  left_join(., 
            AUC, 
            by='Model') %>%
  gather(Criteria, Value, -Model) %>%
  mutate(Criteria = factor(Criteria, 
                           levels = c('Accuracy', 
                                      'MaleAccuracy', 
                                      'FemaleAccuracy', 
                                      'AUC'))) %>%
  ggplot(aes(x=Criteria, y=Value, 
             color=Model, group=Model)) +
  geom_line() +
  geom_point() +
  theme_classic() +
  labs(title='Model Comparison', 
       x=NULL, 
       y=NULL)
  

#TIME====
mean_times <- function(model_result, model){
  df <- 
    model_result %>%
    summarise(fit_time = mean(mean_fit_time, na.rm=T), 
              score_time = mean(mean_score_time, na.rm=T))
  df$Model <- model$Name
  return(df)
}

timesDF <-
  bind_rows(
  mean_times(knn$GSresult, knn), 
  mean_times(knn.pca$GSresult, knn.pca), 
  mean_times(svm$CVresult, svm), 
  mean_times(svm.pca$CVresult, svm.pca), 
  mean_times(dt$CVresult, dt), 
  mean_times(rf$CVresult, rf), 
  mean_times(lr$CVresult, lr), 
  mean_times(lr.norm$CVresult, lr.norm)
) 

fit_time.graph.rep <-
  timesDF %>%
  ggplot(aes(x=reorder(Model, fit_time, desc), 
             y=fit_time, 
             fill=Model)) +
  geom_bar(stat='identity') +
  geom_text(aes(label=Model, 
                y=0), 
            angle=90, 
            hjust=0) +
  theme_classic() +
  theme(axis.text.x = element_blank(), 
        axis.ticks.x = element_blank()) +
  labs(x='Mean Fitting Time', 
       y='Seconds', 
       title='Fitting the Model')

fit_time.graph.pres <-
  timesDF %>%
  ggplot(aes(x=reorder(Model, fit_time, desc), 
             y=fit_time, 
             fill=Model)) +
  geom_bar(stat='identity') +
  geom_text(aes(label=Model, 
                y=fit_time/2)) +
  theme_classic() +
  theme(axis.text.x = element_blank(), 
        axis.ticks.x = element_blank()) +
  labs(x='Mean Fitting Time', 
       y='Seconds', 
       title='Fitting the Model')

score_time.graph.rep <-
  timesDF %>%
  ggplot(aes(x=reorder(Model, score_time, desc), 
             y=score_time, 
             fill=Model)) +
  geom_bar(stat='identity') +
  geom_text(aes(label=Model, 
                y=0), 
            angle=90, 
            hjust=0) +
  theme_classic() +
  theme(axis.text.x = element_blank(), 
        axis.ticks.x = element_blank()) +
  labs(x='Mean Score Time', 
       y='Seconds', 
       title='Scoring with the Model')

score_time.graph.pres <-
  timesDF %>%
  ggplot(aes(x=reorder(Model, score_time, desc), 
             y=score_time, 
             fill=Model)) +
  geom_bar(stat='identity') +
  geom_text(aes(label=Model, 
                y=score_time/2)) +
  theme_classic() +
  theme(axis.text.x = element_blank(), 
        axis.ticks.x = element_blank()) +
  labs(x='Mean Score Time', 
       y='Seconds', 
       title='Scoring with the Model')

#Ensemble ====
ens <- NULL
ens$true <- rf$result$true
ens$rf.pred <- rf$result$pred
ens$knn.pca.pred <- knn.pca$result$pred
ens$svm.pca.pred <- svm.pca$result$pred
ens$lr.pred <- lr$result$pred
ens <- data.frame(ens)

ens.table <-
  ens %>%
  mutate(rf.pred = ifelse(rf.pred == 'male', 1, 0), 
         knn.pca.pred = ifelse(knn.pca.pred == 'male', 1, 0), 
         svm.pca.pred = ifelse(svm.pca.pred == 'male', 1, 0),
         lr.pred = ifelse(lr.pred == 'male', 1, 0),
         groupResult = rf.pred + knn.pca.pred + svm.pca.pred + lr.pred, 
         pred = ifelse(groupResult %in% c(4), 'male', 'female')) %>%
  select(true, pred) %>%
  table(.)

(ens.table[1] + ens.table[4]) / sum(ens.table)
