
# [0] path & packages ---------------------------------------------------------------
path <- c("D:/Google Drive/NCTU/106/下學期/機器學習/HW/Final_Project/dataset/") # Windows
path <- c("/Users/bee/Google Drive/NCTU/106/下學期/機器學習/HW/Final_Project/dataset/") # Mac

library(magrittr)
library(dplyr)
library(ggplot2)
library(jpeg)
library(imager)
#install.packages("drat", repos="https://cran.rstudio.com")
#drat:::addRepo("dmlc")
#install.packages("mxnet")
library(mxnet)



# [1] get row data -----------------------------------------------------------------
par(mfrow=c(2,2))    # set the plotting area into a 1*2 array

pic <- load.image(paste0(path,"/young/female/127.jpg"))
plot(pic,main = "Original")

# shifting
imshift(pic,40,20) %>% plot(main="Shifting")
imshift(pic,runif(1,-20,20),runif(1,-20,20),boundary_conditions=1) %>% plot(main="Shifting")


# blur
blur.layers <- map_il(seq(1,15,l=5),~ isoblur(pic,.))
blur.layers %>% parmin %>% plot(main="Min across blur levels")
blur.layers %>% parmax %>% plot(main="Max across blur levels")
blur.layers %>% average %>% plot(main="Average across blur levels")

# rotation
imrotate(pic,30) %>% plot(main="Rotating 30")
imrotate(pic,-30) %>% plot(main="Rotating -30")


im <- grayscale(pic)
#deriche(im,sigma=4,order=1,axis="y") %>% save.image(.,paste0(path,"/ts.jpg"))

deriche(im,sigma=4,order=1,axis="y") %>% plot(main="1st Deriche of Gaussian along y") 
deriche(im,sigma=4,order=2,axis="y") %>% plot(main="2nd Deriche of Gaussian along y")

vanvliet(im,sigma=4,order=1,axis="y") %>% plot(main="1st Vliet of Gaussian along y")
vanvliet(im,sigma=4,order=2,axis="y") %>% plot(main="2nd Vliet of Gaussian along y")

# -----------------------------------------------------------------------

img1 <- readImg(paste0(path,"/young/female/125.jpg"))
img1$type
img1$dim

imshow(img1)

points(c(300, 530), c(305, 505), col = "red", pch = 19, cex = 2)
lines(c(1, img1$dim[2]), c(1, img1$dim[1]), col = "red", lwd = 8)
lines(c(1, img1$dim[2]), c(img1$dim[1], 1), col = "red", lwd = 8)