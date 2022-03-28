
evaluate <- function(dataPred, dataEval, modelName, setName, levelNumber) {
  dataFilteredEval <- dataEval[dataEval$Argument.ID %in% dataPred$Argument.ID,]
  dataFilteredPred <- dataPred[dataPred$Argument.ID %in% dataFilteredEval$Argument.ID,]
  
  # Remove 'Argument ID'
  dataEval <- dataFilteredEval[,-1]
  # Select only present labels
  dataEval <- dataEval[,names(dataEval) %in% names(dataPred)]
  # Select only labels of current level
  dataPred <- dataFilteredPred[,names(dataEval)]
  
  ## Eval
  
  order <- names(dataPred)
  
  n <- nrow(dataPred)
  m <- length(order)
  
  ma_top_list <- list()
  ma_rec_bottom_list <- list()
  ma_prec_bottom_list <- list()
  correct_Nums_list <- list()
  
  for (j in 1:m) {
    ma_top_list[[j]] <- 0
    ma_rec_bottom_list[[j]] <- 0
    ma_prec_bottom_list[[j]] <- 0
    correct_Nums_list[[j]] <- 0
  }
  
  for (i in 1:n) {
    for (j in 1:m) {
      if (dataEval[i, order[j]] == 1) {
        ma_rec_bottom_list[[j]] <- ma_rec_bottom_list[[j]] + 1
        
        if (dataEval[i, order[j]] == dataPred[i, order[j]]) {
          ma_top_list[[j]] <- ma_top_list[[j]] + 1
          
          ma_prec_bottom_list[[j]] <- ma_prec_bottom_list[[j]] + 1
          
          correct_Nums_list[[j]] <- correct_Nums_list[[j]] + 1
        }
      }
      else if (dataPred[i, order[j]] == 1) {
        ma_prec_bottom_list[[j]] <- ma_prec_bottom_list[[j]] + 1
      } else {
        correct_Nums_list[[j]] <- correct_Nums_list[[j]] + 1
      }
    }
  }
  
  ma_rec_list <- list()
  ma_prec_list <- list()
  ma_f1_list <- list()
  accuracy_list <- list()
  
  ma_rec <- 0
  ma_prec <- 0
  sum_acc <- 0
  for (j in 1:m) {
    ma_rec_list[[j]] <- ifelse(ma_rec_bottom_list[[j]] == 0, 0, ma_top_list[[j]] / ma_rec_bottom_list[[j]])
    ma_rec <- ma_rec + ma_rec_list[[j]]
    ma_prec_list[[j]] <- ifelse(ma_prec_bottom_list[[j]] == 0, 0, ma_top_list[[j]] / ma_prec_bottom_list[[j]])
    ma_prec <- ma_prec + ma_prec_list[[j]]
    sum <- ma_prec_list[[j]] + ma_rec_list[[j]]
    ma_f1_list[[j]] <- ifelse(sum == 0, 0.0,  2 * (ma_prec_list[[j]] * ma_rec_list[[j]]) / (sum))
    accuracy_list[[j]] <- correct_Nums_list[[j]] / n
    sum_acc <- sum_acc + accuracy_list[[j]]
  }
  
  ma_rec <- ma_rec / m
  ma_prec <- ma_prec / m
  ma_f1 <- 2 * (ma_prec * ma_rec) / (ma_prec + ma_rec)
  sum_acc <- sum_acc / m
  
  Label <- c(order, "Mean")
  
  output <- data.frame(Label, stringsAsFactors = FALSE)
  output$Precision <- c(unlist(ma_prec_list), ma_prec)
  output$Recall <- c(unlist(ma_rec_list), ma_rec)
  output$F1 <- c(unlist(ma_f1_list), ma_f1)
  output$Accuracy <- c(unlist(accuracy_list), sum_acc)
  
  output$Method <- c(modelName)
  output$`Test dataset` <- c(setName)
  output$Level <- c(levelNumber)
  
  output <- output[, c('Method', 'Test dataset', 'Level', 'Label', 'Precision', 'Recall', 'F1', 'Accuracy')]
  
  return (output)
}
