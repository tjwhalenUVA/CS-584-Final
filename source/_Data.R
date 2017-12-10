mainFolder = 'C:/Users/e481340/Documents/GMU MASTERS/CS 584/CS584_Final'

voice <- read_csv(paste(mainFolder, 
                        "/data/voice.csv", 
                        sep = "")
)

definitions <- read_excel(paste(mainFolder, 
                                "/data/definitions.xlsx", 
                                sep = "")
)

#KNN====
knn <- NULL
knn$Name <- "K-Nearest Neighbors"

knn$params <- read_excel(paste(mainFolder, 
                              "/results/knn_results.xlsx", 
                              sep = ""), 
                        sheet = "params"
)

knn$result <- read_excel(paste(mainFolder, 
                               "/results/knn_results.xlsx", 
                               sep = ""), 
                         sheet = "result"
)

knn$GSresult <- read_excel(paste(mainFolder, 
                               "/results/knn_results.xlsx", 
                               sep = ""), 
                         sheet = "GSresult"
)

knn$roc <- read_excel(paste(mainFolder, 
                                 "/results/knn_results.xlsx", 
                                 sep = ""), 
                           sheet = "roc"
)

knn$auc <- read_excel(paste(mainFolder, 
                            "/results/knn_results.xlsx", 
                            sep = ""), 
                      sheet = "auc"
)

#DT====
dt <- NULL

dt$Name <- "Decision Tree"

dt$params <- read_excel(paste(mainFolder, 
                               "/results/dt_results.xlsx", 
                               sep = ""), 
                         sheet = "params"
)

dt$result <- read_excel(paste(mainFolder, 
                               "/results/dt_results.xlsx", 
                               sep = ""), 
                         sheet = "result"
)

dt$tree <- read_excel(paste(mainFolder, 
                              "/results/dt_results.xlsx", 
                              sep = ""), 
                        sheet = "tree"
)

dt$roc <- read_excel(paste(mainFolder, 
                            "/results/dt_results.xlsx", 
                            sep = ""), 
                      sheet = "roc"
)

dt$auc <- read_excel(paste(mainFolder, 
                           "/results/dt_results.xlsx", 
                           sep = ""), 
                     sheet = "auc"
)

#RF====
rf <- NULL

rf$Name <- "Random Forest"

rf$params <- read_excel(paste(mainFolder, 
                              "/results/rf_results.xlsx", 
                              sep = ""), 
                        sheet = "params"
)

rf$result <- read_excel(paste(mainFolder, 
                              "/results/rf_results.xlsx", 
                              sep = ""), 
                        sheet = "result"
)

rf$roc <- read_excel(paste(mainFolder, 
                              "/results/rf_results.xlsx", 
                              sep = ""), 
                        sheet = "roc"
)

rf$auc <- read_excel(paste(mainFolder, 
                           "/results/rf_results.xlsx", 
                           sep = ""), 
                     sheet = "auc"
)

rf$CVresult <- read_excel(paste(mainFolder, 
                           "/results/rf_results.xlsx", 
                           sep = ""), 
                     sheet = "CVresult"
)

#SVM====
svm <- NULL

svm$Name <- "Support Vector Machine"

svm$params <- read_excel(paste(mainFolder, 
                              "/results/svm_results.xlsx", 
                              sep = ""), 
                        sheet = "params"
)

svm$result <- read_excel(paste(mainFolder, 
                              "/results/svm_results.xlsx", 
                              sep = ""), 
                        sheet = "result"
)

svm$roc <- read_excel(paste(mainFolder, 
                               "/results/svm_results.xlsx", 
                               sep = ""), 
                         sheet = "roc"
)

svm$auc <- read_excel(paste(mainFolder, 
                            "/results/svm_results.xlsx", 
                            sep = ""), 
                      sheet = "auc"
)

svm$CVresult <- read_excel(paste(mainFolder, 
                            "/results/svm_results.xlsx", 
                            sep = ""), 
                      sheet = "CVresult"
)

#KNN PCA====
knn.pca <- NULL

knn.pca$Name <- "K-Nearest Neighbors (PCA)"

knn.pca$params <- read_excel(paste(mainFolder, 
                               "/results/knn_pca_results.xlsx", 
                               sep = ""), 
                         sheet = "params"
)

knn.pca$result <- read_excel(paste(mainFolder, 
                               "/results/knn_pca_results.xlsx", 
                               sep = ""), 
                         sheet = "result"
)

knn.pca$GSresult <- read_excel(paste(mainFolder, 
                                   "/results/knn_pca_results.xlsx", 
                                   sep = ""), 
                             sheet = "GSresult"
)

knn.pca$roc <- read_excel(paste(mainFolder, 
                                     "/results/knn_pca_results.xlsx", 
                                     sep = ""), 
                               sheet = "roc"
)

knn.pca$auc <- read_excel(paste(mainFolder, 
                                "/results/knn_pca_results.xlsx", 
                                sep = ""), 
                          sheet = "auc"
)

#SVM PCA====
svm.pca <- NULL

svm.pca$Name <- "Support Vector Machine (PCA)"

svm.pca$params <- read_excel(paste(mainFolder, 
                               "/results/svm_pca_results.xlsx", 
                               sep = ""), 
                         sheet = "params"
)

svm.pca$result <- read_excel(paste(mainFolder, 
                               "/results/svm_pca_results.xlsx", 
                               sep = ""), 
                         sheet = "result"
)

svm.pca$roc <- read_excel(paste(mainFolder, 
                                   "/results/svm_pca_results.xlsx", 
                                   sep = ""), 
                             sheet = "roc"
)

svm.pca$auc <- read_excel(paste(mainFolder, 
                                "/results/svm_pca_results.xlsx", 
                                sep = ""), 
                          sheet = "auc"
)

svm.pca$CVresult <- read_excel(paste(mainFolder, 
                                 "/results/svm_pca_results.xlsx", 
                                 sep = ""), 
                           sheet = "CVresult"
)


#Log Regression====
lr <- NULL

lr$Name <- "Logistic Regression"

lr$params <- read_excel(paste(mainFolder, 
                                   "/results/lr_results.xlsx", 
                                   sep = ""), 
                             sheet = "params"
)

lr$result <- read_excel(paste(mainFolder, 
                                   "/results/lr_results.xlsx", 
                                   sep = ""), 
                             sheet = "result"
)

lr$roc <- read_excel(paste(mainFolder, 
                                "/results/lr_results.xlsx", 
                                sep = ""), 
                          sheet = "roc"
)

lr$auc <- read_excel(paste(mainFolder, 
                           "/results/lr_results.xlsx", 
                           sep = ""), 
                     sheet = "auc"
)

lr$CVresult <- read_excel(paste(mainFolder, 
                                 "/results/lr_results.xlsx", 
                                 sep = ""), 
                           sheet = "CVresult"
)



#Log Regression (PCA)====
lr.norm <- NULL

lr.norm$Name <- "Logistic Regression (Normal)"

lr.norm$params <- read_excel(paste(mainFolder, 
                              "/results/lr_pca_results.xlsx", 
                              sep = ""), 
                        sheet = "params"
)

lr.norm$result <- read_excel(paste(mainFolder, 
                              "/results/lr_pca_results.xlsx", 
                              sep = ""), 
                        sheet = "result"
)

lr.norm$roc <- read_excel(paste(mainFolder, 
                           "/results/lr_pca_results.xlsx", 
                           sep = ""), 
                     sheet = "roc"
)

lr.norm$auc <- read_excel(paste(mainFolder, 
                                "/results/lr_pca_results.xlsx", 
                                sep = ""), 
                          sheet = "auc"
)

lr.norm$CVresult <- read_excel(paste(mainFolder, 
                                "/results/lr_pca_results.xlsx", 
                                sep = ""), 
                          sheet = "CVresult"
)
