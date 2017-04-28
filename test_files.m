%% test file
clc
clear
close all
path = 'F:\opencvjpg\' ;
file_name = '19.jpg' ;
file_path = [path, file_name] ;
img = imread(file_path) ;

[plate_cell, plate_img,chars_center ,plate_type, score] = recognizePlate(img) ;
for i = 1:length(plate_cell)
    subplot(1,length(plate_cell),i)
    imshow(plate_cell{i})
end





