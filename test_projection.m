clc
close all
clear all
img = imread('F:\opencvjpg\1016.jpg') ;
imshow(img)
p1 = ginput(4) ;
p2 = [100,100;280,100;100,160;280,160] ;
T = calc_homography(p1,p2); 
% [h1 w1]=size(img);
T = maketform('projective',T);   %投影矩阵
[imgn X Y]=imtransform(img,T);     %投影
figure
subplot(1,2,1)
imshow(img)
subplot(1,2,2)
imshow(imgn)




