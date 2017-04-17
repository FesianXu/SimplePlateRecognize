%%%%
% Author: FesianXu
% 全自动提取图片，调用recognizePlate库函数（自己编写）
%%%%
clc
clear all
close all
%% get a frame of image
path = 'F:\opencvjpg\' ;
file_name = '1109.jpg' ; 
save_path = 'F:\opencvjpg\training sets\save_raw\' ;
%%% 1014 1016 1026 big problem, 71 is too gray 1071 addressed !
%%% 1043 34 71 multiple test addressed!
%%% 1080 1120 regetchar failed, the cell have been over 8 list addressed!
%%% 1016 imgNormal failed
%%% 74 refine failed
%%% 1014 汉字定位问题
%%% 在车牌分辨率比较低的时候可能会出现一些小问题 49
%%% a little problem in chars segment in 1110
%%% 1016 分割效果不是特别好
% file_path = [path, file_name] ;
% img_color = imread(file_path) ;
% [plate_cell, plate_type, score] = recognizePlate(img_color) ;
% for i = 1:length(plate_cell)
%     subplot(1,length(plate_cell),i)
%     imshow(plate_cell{i})
% end


dirimg = dir(path) ;
invalid = 0 ;
valid = 0 ;
warning('off')
for j = 3:173
    fprintf('第%d张图片\r\n',j-2) ;
    file_name = [path,dirimg(j).name] ;
    img_color = imread(file_name) ;
    try
        [plate_cell, plate_type, score] = recognizePlate(img_color) ;
    catch
        invalid = invalid+1 ;
        continue ;
    end
    if isempty(plate_cell{1})
        invalid = invalid+1 ;
        continue ;
    end
    valid = valid+1 ;
    for i = 1:length(plate_cell)
        save_name = [save_path, num2str(j-2),'--',num2str(i),'.jpg'];
        imwrite(plate_cell{i},save_name) ;
    end
end
fprintf('有效图片数%d\r\n',valid) ;
fprintf('无效图片数%d\r\n',invalid) ;
fprintf('图片总数%d\r\n',valid+invalid);



