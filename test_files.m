%% test file
clc
clear
close all
path = 'F:\opencvjpg\' ;
file_name = '1069.jpg' ;
file_path = [path, file_name] ;
img = imread(file_path) ;

[plate_cell, plate_img,chars_center ,plate_type, score] = recognizePlate(img) ;
imshow(plate_img)




