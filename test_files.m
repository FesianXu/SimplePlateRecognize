%% test file
clc
clear
close all
path = 'F:\opencvjpg\' ;
auto_reg_save_path = '.\res\auto_samples_set\' ;
file_name = '1197.jpg' ;
file_path = [path, file_name] ;
img = imread(file_path) ;

[plate_cell, plate_img,chars_center ,correct_type, plate_type, score] = recognizePlate(img) ;
for i = 1:length(plate_cell)
    subplot(1,length(plate_cell),i)
    imshow(plate_cell{i})
    title(getCharName(plate_type(i)))
end




