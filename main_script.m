%%%
% Author:FesianXu
% the teaching demo of plate detection
% version1.0: simple function, not affine transform, not machine learning
% algorithms,not smart char divide algorithms
% version1.1: add projective transform to refine the plate ,with simple
% recognization technique.
% version1.2: add a better method to find the four corner in plate to help
% projective transform, and add a method to delete the frame around the
% plate
% version1.3: add plate normalization .
%%%

clc
clear
close all
%% parameter initiation
%%% 一些参数的初始化和设定
char_nor_width = 16 ;
char_nor_height = 32 ;
plate_nor_width = 160 ;
plate_nor_height = 48 ;
plate_wh_upper = 4 ; % 车牌长宽比例上界
plate_wh_lower = 1 ; % 车牌长宽比下界
plate_wh_atlease_width = 30 ; % 车牌至少长度
plate_wh_atlease_height = 30 ; % 车牌至少宽度
plate_area_norm_width = 250 ; % 车牌假设域长度
plate_area_norm_height = 100 ; % 车牌假设域宽度
char_wh_upper = 4 ; % 字符长宽比上界
char_wh_lower = 1 ; % 字符长宽比下界
plate_area_width_bias = 10 ; % 车牌假设区域长度偏移
plate_area_height_bias = 10 ; % 车牌假设区域宽度偏移
img_resize_width = 800 ;
img_resize_height = 600 ;
%%% 先归一化到一个中等的尺度，并且遍历可能的车牌区域
%%% 然后在较大尺度中提取出车牌域进行下一步处理。

%% get image and turn it to hsv space
%%% 根据先验知识，提取出车牌区域，需要注意的是，需要排除一些明显的非车牌域
%%% 因为是读图片，而不是读视频，所以不需要做动态模糊处理。
path = 'F:\opencvjpg\' ;
file_name = '1014.jpg' ; % 1014 1016
file_path = [path, file_name] ;
img_color = imread(file_path) ;
img_color_resize = imresize(img_color,[img_resize_height,img_resize_width]) ;
% img_gray_resize = rgb2gray(img_color_resize) ;
img_merge = getBluePlate(img_color_resize) ; %%% get blue area and mask blue area with pixel 1
erode_core = ones(2,2) ;
img_merge = imerode(img_merge, erode_core) ;
dilate_core = ones(10,10) ;
img_merge = imdilate(img_merge, dilate_core) ;
%%% 进行形态学闭操作，得出初步车牌目标二值图
img_merge_con = bwboundaries(img_merge,8, 'noholes') ; %% 8-连通域检测
len_img_merge_con = length(img_merge_con) ;

%% search for plate location may be need to judge if it is really plate,delete some area
%%% 删除掉一些明显不是车牌假设区域的，此时车牌可能会存在仿射变换或者透视变换，需要进一步
%%% 进行甄别
inner_loop = 1 ;
save_con = {} ; % 保存经过车牌尺寸排查的车牌假设域
for i = 1:len_img_merge_con
    drow = max(img_merge_con{i}(:,1))-min(img_merge_con{i}(:,1)) ;
    dcol = max(img_merge_con{i}(:,2)-min(img_merge_con{i}(:,2))) ;
    if dcol/drow < plate_wh_upper && dcol/drow > plate_wh_lower && dcol > plate_wh_atlease_width && drow > plate_wh_atlease_height
        save_con{inner_loop} = img_merge_con{i} ;
        inner_loop = inner_loop+1 ;
    end
end

%% here we need to project transform to refine the plates
%%% 提取出车牌假设区域的四个边角，映射到新区域中，做透视变换，矫正其倾斜。
%%% 采用角度特征法

if isempty(save_con)
    disp('save con is empty!') ;
    return ;
end

[pl_img, ~] = extractPlate(img_color_resize, save_con) ;
pl_norm_img = cell(length(pl_img),1) ;
for i =1:length(pl_img)
    pl_norm_img{i} = imresize(pl_img{i}, [plate_area_norm_height, plate_area_norm_width]) ;
end %% 标准化车牌假设域

%%% 取其中一个做测试
img_test = pl_norm_img{1} ;
imgt_merge = getBluePlate(img_test) ;
img_dilate_core = ones(5,5) ;
imgt_merge = imdilate(imgt_merge, img_dilate_core) ;
imgt_con = bwboundaries(imgt_merge,8, 'noholes') ;
len_imgt_con = length(imgt_con) ;
inner_loop = 1 ;
imgt_save_con = {} ;
for i = 1:len_imgt_con
    drow = max(imgt_con{i}(:,1))-min(imgt_con{i}(:,1)) ;
    dcol = max(imgt_con{i}(:,2))-min(imgt_con{i}(:,2)) ;
    if dcol/drow < plate_wh_upper && dcol/drow > plate_wh_lower && dcol > 20 && drow > 20
        imgt_save_con{inner_loop} = imgt_con{i} ;
        inner_loop = inner_loop+1 ;
    end
end
%%% 车牌域特写
if isempty(imgt_save_con)
    disp('imgt save con is empty');
    return ;
end

%% 捕获角点特征矫正车牌
%%% 矫正车牌
list = imgt_save_con{1} ;
[points,left,right] = getPlateCorner(list) ;
p11 = points(1,:) ;
p12 = points(2,:) ;
p21 = points(3,:) ;
p22 = points(4,:) ;
p11 = p11(end:-1:1) ;
p12 = p12(end:-1:1) ;
p21 = p21(end:-1:1) ;
p22 = p22(end:-1:1) ;
area_new = [50,50;50+plate_nor_width,50;50,50+plate_nor_height;50+plate_nor_width,50+plate_nor_height] ;
area_old = [p11;p12;p21;p22] ;
Tran = calc_homography(area_old,area_new) ;
Tran = maketform('projective',Tran);   %投影矩阵
[imgn, X, Y] = imtransform(img_test,Tran);     %投影

%% 矫正测试
% figure(3)
% subplot(2,1,1)
% imshow(img_test)
% hold on
% plot(p11(1,1),p11(1,2),'g*')
% plot(p12(1,1),p12(1,2),'r*')
% plot(p21(1,1),p21(1,2),'b*')
% plot(p22(1,1),p22(1,2),'y*')
% hold on
% plot(list(1:30,2),list(1:30,1),'y*')
% hold on
% plot(left(:,2),left(:,1),'r*')
% plot(right(:,2),right(:,1),'b*')
% subplot(2,1,2)
% imshow(imgn)
% figure(4)
% imshow(imgt_merge)


%% 删除车牌边框
imgn_gray = rgb2gray(imgn) ;
bw_thres = 0.5 ;
imgn_bw = im2bw(imgn_gray,bw_thres) ;
imgn_out = deletePlateFrame(imgn_bw) ;
erode_core = ones(2,2) ;
imgn_out = imerode(imgn_out, erode_core) ;
imshow(imgn_out)

%% get chars in plate
%%% 车牌字符分割
con = bwboundaries(imgn_out,8, 'noholes') ;














