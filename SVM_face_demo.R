
# -- Machine Learning 2017 -- 
# [Final project] : CNN for human face classification
# [Deadline] : 2017/6/19

#====================================================================================
# Functions
#====================================================================================

read_rawdata <- function(n,age,gender,dim){
  M <- sapply(1:n, function(i) load.image(paste0(path,age,"/",gender,"/",i,".jpg")) %>% 
                #grayscale %>% 
                #resize(.,size_x = dim, size_y = dim) %>%
                as.vector() )
  M <- M %>% t %>% cbind.data.frame(class=paste0(age,"_",gender),.)
  return(M)
}

get_error_rate <- function(data,n){
  
  hit_table <- table(data[,c(1,n)])
  
  hit_rate <- sum(diag(hit_table)) / nrow(data)
  
  error_rate <- 1 - hit_rate
  
  print(paste0("Error rate of ",colnames(data)[n]," : ",error_rate))
}

#====================================================================================
# Main 
#====================================================================================

# [0] path & packages ---------------------------------------------------------------
path <- c("D:/Google Drive/NCTU/106/下學期/機器學習/HW/Final_Project/dataset_face/") # Windows
path <- c("/Users/bee/Google Drive/NCTU/106/下學期/機器學習/HW/Final_Project/dataset_face/") # Mac

library(magrittr)
library(dplyr)
library(ggplot2)
library(jpeg)
library(imager)
library(e1071)


# [1] get row data -----------------------------------------------------------------
dim <- 64
tic <- proc.time()

# combine all categories
train_raw <- rbind.data.frame(read_rawdata(1588,"adult", "male",  dim ),
                              read_rawdata(1210,"adult", "female",dim ),
                              read_rawdata(650, "child", "male",  dim ),
                              read_rawdata(774, "child", "female",dim ),
                              read_rawdata(878, "elder", "male",  dim ),
                              read_rawdata(452, "elder", "female",dim ),
                              read_rawdata(830, "young", "male",  dim ),
                              read_rawdata(1002,"young", "female",dim ))

levels(train_raw$class) <- list( "0"="child_male",
                                 "1"="young_male",
                                 "2"="adult_male",
                                 "3"="elder_male",
                                 "4"="child_female",
                                 "5"="young_female",
                                 "6"="adult_female",
                                 "7"="elder_female")
print(proc.time() - tic)


# [2] data seperation ---------------------------------------------------------------

# Seperate raw data to training/testing data
set.seed(9487)


X_train <- train_raw[,-1] 
T_train <- train_raw[,1] 

# for unbalance data, we need to adjust class weight
class_weight <- (1000/table(T_train)) 
class_weight

# [3] C-SVM model -------------------------------------------------
memory.limit(size=56000)


C10_SVM_radial_weight <- svm(X_train, 
                             T_train, 
                             type = "C-classification", 
                             kernel = "radial",
                             class.weights =	class_weight,
                             cost =  10)

# [4] demo ---------------------------------------------------------

demo_path <- c("D:/Google Drive/NCTU/106/下學期/機器學習/HW/Final_Project/dataset_face/young/female/")
demo <- sapply(0:639, function(i) load.image(paste0(demo_path,i,".jpg")) %>% as.vector)
demo <- demo %>% t 

demo_output <- predict(C10_SVM_radial_weight, demo)
table(demo_output)

result <- data.frame(ID = sapply(0:639, function(i) paste0(i,".jpg")),
                     op = demo_output)
write.table(result,
            file = paste0(demo_path,"result.csv"),col.names = F,row.names = F,sep = ",")
