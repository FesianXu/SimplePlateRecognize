%% test file
clc
clear 
close all

path = 'F:\opencvjpg\';
file_name = '1140.jpg' ;
file_path = [path,file_name] ;
img = imread(file_path) ;
imghsv = rgb2hsv(img) ;
V = imghsv(:,:,3) ;
Veq = histeq(V) ;
subplot(1,2,1)
imshow(V)
subplot(1,2,2)
imshow(Veq)
can = edge(V,'canny');
figure
imshow(can)



