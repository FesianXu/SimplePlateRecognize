%% test file
clc
clear all
close all
path = 'F:\opencvjpg\' ;
file_name = '40.jpg' ;
file_path = [path, file_name] ;
img = imread(file_path) ;
figure(1)
subplot(3,1,1)
imshow(img) ;
R = img(:,:,1) ;
G = img(:,:,2) ;
B = img(:,:,3) ;
a1 = 0.05 ;
a2 = 0.05 ;
a3 = 1-a1-a2 ;
gray_img = a1*R+a2*G+a3*B ;
subplot(3,1,2)
imshow(gray_img) 
img_gray = rgb2gray(img) ;
subplot(3,1,3)
imshow(img_gray)

figure(2)
th = graythresh(gray_img) ;
imgbw = im2bw(gray_img, 0.8) ;
imshow(imgbw)




