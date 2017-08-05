# -- Machine Learning 2017 -- 
# [Final project] : CNN for human face classification
# [Deadline] : 2017/6/19

#====================================================================================
# Functions
#====================================================================================

# read all 'jpg' from a folder, and get a list of RGB matrix ----
get_data <- function(n,age,gender){
  lapply(0:n, function(x) readJPEG(paste0(path,age,"/",gender,"/",x,".jpg")))
}

# read all 'jpg' from a folder, and convert to GrayScale and downsize to 30x30 ----
get_rawdata <- function(n,age,gender,dim){
  M <- matrix(0,1,dim**2) # 30x30 pixels
  for(i in 0:n){
    tmp <- load.image(paste0(path,age,"/",gender,"/",i,".jpg")) %>%
      grayscale() %>%
      resize(.,size_x = dim, size_y = dim) %>%
      as.vector() 
    M <- rbind(M,tmp)
  }
  
  M <- as.data.frame(M)
  M <- data.frame(Age = rep(age,n+1),
                  Gender = rep(gender,n+1),
                  num = 0:n) %>% cbind.data.frame(.,M[-1,])
  return(M)
}

#====================================================================================
# Main 
#====================================================================================

# [0] path & packages ---------------------------------------------------------------
path <- c("D:/Google Drive/NCTU/106/ä¸?å­¸æ??/æ©???¨å­¸ç¿?/HW/Final_Project/dataset/") # Windows
path <- c("/Users/bee/Google Drive/NCTU/106/ä¸?å­¸æ??/æ©???¨å­¸ç¿?/HW/Final_Project/dataset/") # Mac

library(magrittr)
library(dplyr)
library(ggplot2)
library(jpeg)
library(imager)
library(mxnet)

# [1] get row data -----------------------------------------------------------------
# A : adult     0 M : male
# C : child     1 F : famale
# E : elder
# Y : young
dim <- 64
tic <- proc.time()

A_M <- get_rawdata(1587,"adult","male",dim )
A_F <- get_rawdata(1209,"adult","female",dim )

C_M <- get_rawdata(649,"child","male",dim )
C_F <- get_rawdata(773,"child","female",dim )

E_M <- get_rawdata(877,"elder","male",dim )
E_F <- get_rawdata(451,"elder","female",dim)

Y_M <- get_rawdata(829,"young","male",dim )
Y_F <- get_rawdata(1001,"young","female",dim )
# --------------------------------------------------------------
A_M <- read_rawdata(1587,"adult","male",dim )
A_F <- read_rawdata(1209,"adult","female",dim )

C_M <- read_rawdata(649,"child","male",dim )
C_F <- read_rawdata(773,"child","female",dim )

E_M <- read_rawdata(877,"elder","male",dim )
E_F <- read_rawdata(451,"elder","female",dim)

Y_M <- read_rawdata(829,"young","male",dim )
Y_F <- read_rawdata(1001,"young","female",dim )

print(proc.time() - tic)

# combine all categories
train_raw <- rbind.data.frame(A_M,A_F,
                              C_M,C_F,
                              E_M,E_F,
                              Y_M,Y_F)
levels(train_raw$Gender) <- list("0"="male", "1"="female")

# [2] RGB to Gray scale ----------------------------------------------------

# [3] make Size consistent -------------------------------------------------

# [4] CNN model ---------------------------------------------------------------

# Seperate raw data to training/testing data
  set.seed(9487948)
  testing_index <- sample(1:nrow(train_raw),nrow(train_raw) %/% 10 )
  
  training_data <- train_raw[-testing_index,]
  testing_data <- train_raw[testing_index,]

# [Gender Classify]
 train.x <- training_data[,-c(1:3)] 
 train.y <- training_data[,2] %>% as.character() %>% as.numeric()
 

 test <- testing_data[,-c(1:3)]
 table(train.y)



# [5] testing data ----------------------------------------------------------------------------
 # Convolutional NN
  data <- mx.symbol.Variable('data')
  # first conv
  conv1 <- mx.symbol.Convolution(data=data, kernel=c(5,5), num_filter=20)
  tanh1 <- mx.symbol.Activation(data=conv1, act_type="tanh")
  pool1 <- mx.symbol.Pooling(data=tanh1, pool_type="max",kernel=c(2,2), stride=c(2,2))
  # second conv
  conv2 <- mx.symbol.Convolution(data=pool1, kernel=c(5,5), num_filter=50)
  tanh2 <- mx.symbol.Activation(data=conv2, act_type="tanh")
  pool2 <- mx.symbol.Pooling(data=tanh2, pool_type="max",kernel=c(2,2), stride=c(2,2))
  # first fullc
  flatten <- mx.symbol.Flatten(data=pool2)
  fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=500)
  tanh3 <- mx.symbol.Activation(data=fc1, act_type="tanh")
  # second fullc
  fc2 <- mx.symbol.FullyConnected(data=tanh3, num_hidden=10)
  # loss
  lenet <- mx.symbol.SoftmaxOutput(data=fc2)
    
  train.array <- train.x %>% t
  dim(train.array) <- c(dim, dim, 1, nrow(train.x))
  
  test.array <- test %>% t
  dim(test.array) <- c(dim, dim, 1, nrow(test))
  
  mx.set.seed(0)
  
  devices <- mx.cpu()
  
  tic <- proc.time()
  model <- mx.model.FeedForward.create(lenet, X=train.array, y=train.y,
                                      ctx=devices, num.round=75, array.batch.size=100,
                                       learning.rate=0.05, momentum=0.9, wd=0.00001,
                                       eval.metric=mx.metric.accuracy,
                                       epoch.end.callback=mx.callback.log.train.metric(100))


  print(proc.time() - tic)
  
  preds <- predict(model, test.array)
  pred.label <- max.col(t(preds)) - 1
  hit_table <- table(testing_data[,2],pred.label)
  hit_table
  sum(diag(hit_table)) / sum(hit_table)
   
# code test------------------------------------------------------
dt <- load.image(paste0(path,"adult/male/",1,".jpg"))

dt_gray <- grayscale(dt) 
dt_resize <- resize(dt_gray,
                    size_x = 30,
                    size_y = 30)

dt_array <- dt_resize %>% as.vector()
A_M <- rbind(A_M, dt_array)

dt37 <- load.image(paste0(path,"adult/male/",37,".jpg"))
dt37_gray <- grayscale(dt37) 

plot(dt)
plot(dt37_gray)
plot(dt_resize)
#--------------------------------------------------------
