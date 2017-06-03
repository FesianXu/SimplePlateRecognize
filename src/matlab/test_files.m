%% test file
clc
clear
close all
path = 'F:\opencvjpg\new_plate_img\' ;
auto_reg_save_path = '..\..\res\auto_samples_set\' ;
file_name = '40.jpg' ;
file_path = [path, file_name] ;
% [file_name_ui, path_name_ui] = uigetfile('F:\opencvjpg\*.jpg','Select Image') ;
% img = imread([path_name_ui, file_name_ui]) ;
img = imread(file_path) ;

tic ;
[plate_cell, plate_img,chars_center ,correct_type, plate_type, score] = recognizePlate(img) ;
figure
for i = 1:length(plate_cell)
    subplot(1,length(plate_cell),i)
    imshow(plate_cell{i})
    title(getCharName(plate_type(i)))
end

toc ;


