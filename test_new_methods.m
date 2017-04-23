%% test file
clc
clear all
close all
path = 'F:\opencvjpg\' ;
file_name = '40.jpg' ;
file_path = [path, file_name] ;
img = imread(file_path) ;
imgg = rgb2gray(img) ;
sobel_core = [1 2 1;0 0 0;-1 -2 -1] ;
img2 = imfilter(imgg,sobel_core) ;
imshow(img2)
