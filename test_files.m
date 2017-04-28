%% test file
clc
clear
close all
path = 'F:\opencvjpg\test_img\' ;
auto_reg_save_path = '.\res\auto_samples_set\' ;
file_name = '43.jpg' ;
file_path = [path, file_name] ;
img = imread(file_path) ;

[plate_cell, plate_img,chars_center ,correct_type, plate_type, score] = recognizePlate(img) ;
for i = 1:length(plate_cell)
    subplot(1,length(plate_cell),i)
    imshow(plate_cell{i})
    title(getCharName(plate_type(i)))
end

saveRecognizeChars(plate_cell, plate_type, auto_reg_save_path) ;



