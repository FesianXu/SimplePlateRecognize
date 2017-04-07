%%%
% Author:FesianXu
% the teaching demo of plate detection
% version1.0: simple function, not affine transform, not machine learning
% algorithms,not smart char divide algorithms
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
plate_wh_lower = 1.2 ; % 车牌长宽比下界
char_wh_upper = 2.5 ; % 字符长宽比上界
char_wh_lower = 1.2 ; % 字符长宽比下界
plate_area_width_bias = 10 ; % 车牌假设区域长度偏移
plate_area_height_bias = 10 ; % 车牌假设区域宽度偏移

%% get image and turn it to hsv space
%%% 进行一些预处理之后，根据先验知识初步提取出车牌假设区域。
path = 'F:\opencvjpg\' ;
file_name = '41.jpg' ;
file_path = [path, file_name] ;
img = imread(file_path) ;
merge = getBluePlate(img) ;
% get hsv blue
core = ones(10,10) ;
merge = imdilate(merge, core) ;
subplot(2,2,1)
imshow(img)
subplot(2,2,2)
imshow(merge)
con = bwboundaries(merge,8, 'noholes') ;
con_size = size(con) ;
con_size = con_size(1,1) ;
save_con = {} ;
inner = 1 ;

%% search for plate location may be need to judge if it is really plate,delete some area
%%% 删除掉一些明显不是车牌假设区域的，此时车牌可能会存在仿射变换或者透视变换，需要进一步
%%% 进行甄别
for i = 1:con_size
    dy = max(con{i}(:,1))-min(con{i}(:,1)) ;
    dx = max(con{i}(:,2)-min(con{i}(:,2))) ;
    if dx/dy < plate_wh_upper && dx/dy > plate_wh_lower && dx > 40 && dy > 30
        save_con{inner} = con{i} ;
        inner = inner+1 ;
    end
end

%% here we need to project transform to refine the plates
%%% 提取出车牌假设区域的四个特征点，映射到新区域中，做透视变换，矫正其倾斜。
%%% 连续求导自迭代法
[eximg, ~] = extractPlate(merge, save_con) ; % extract plate area
eximg = imgNormal(eximg, plate_nor_width, plate_nor_height) ;
list = save_con{1} ;
col_gap = 30 ;
row_step = 30 ;
k = 0 ;
delta = [] ;
list_new = [list; list(1:row_step,:)] ; % 循环后缀
list_len = length(list(:,1)) ;
for i = 1:list_len
    delta(i) = (list_new(i,1)-list_new(i+row_step,1)) ;
    if i == 1370
        subplot(2,2,2)
        hold on
        plot(list_new(i,2),list_new(i,1),'r*')
    end
    if i == 5+row_step
        subplot(2,2,2)
        hold on
        plot(list_new(i,2),list_new(i,1),'b*')
    end
    if i == 140
        subplot(2,2,2)
        hold on
        plot(list_new(i,2),list_new(i,1),'g*')
    end
    if i == 154+row_step
        subplot(2,2,2)
        hold on
        plot(list_new(i,2),list_new(i,1),'r*')
    end
end
delta_len = length(delta) ;
del2 = [] ;
for i = 1:delta_len-1
    del2(i) = delta(i+1)-delta(i) ;
end
delta_m = abs(delta) >= row_step ;
del2_m = del2 == 0 ;
del2_m(298) = 0; 

figure(5)
subplot(2,1,1)
stem(delta_m,'r*')
subplot(2,1,2)
stem(del2_m,'r*')
figure(6)
tina = delta_m.*del2_m ;
tina_len = length(tina) ;
tii = [] ;
for i = 1:tina_len-1
    tii(i) = tina(i+1)-tina(i) ;
end
stem(tii) ;
a = 0 ;
% row_max = max(list(:,1));
% row_min = min(list(:,1));
% col_max = max(list(:,2)) ;
% col_min = min(list(:,2)) ;
% col_min_aroud = abs(col_min-list(:,2)) < 15;
% col_min_list = list(col_min_aroud == 1,1) ;
% col_min_row_min = min(col_min_list) ;
% 
% col_max_around = abs(col_max-list(:,2)) < 15 ;
% col_max_list = list(col_max_around == 1, 1) ;
% col_max_row_max = max(col_max_list) ;
% % point_11 = [col_min_row_min, col_min] ;
% % point_12 = [row_min,col_max] ;
% % point_21 = [row_max,col_min] ;
% % point_22 = [col_max_row_max, col_max] ;
% point_11 = [col_min, col_min_row_min] ;
% point_12 = [col_max,row_min] ;
% point_21 = [col_min,row_max] ;
% point_22 = [col_max, col_max_row_max] ;
% area_old = [point_11;point_12;point_21;point_22] ; % 列优先排序
% 
% figure
% imshow(img)
% hold on
% plot(list(:,2),list(:,1),'r*')
% plot(point_11(1,1),point_11(1,2), 'r*')
% plot(point_12(1,1),point_12(1,2), 'r*')
% plot(point_21(1,1),point_21(1,2), 'r*')
% plot(point_22(1,1),point_22(1,2), 'r*')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% col_bias = 100 ;
% pointnew_11 = [col_min_row_min, col_min] ;
% pointnew_12 = [col_min_row_min, col_min+plate_nor_width] ;
% pointnew_21 = [col_min_row_min+plate_nor_height, col_min] ;
% pointnew_22 = [col_min_row_min+plate_nor_height, col_min+plate_nor_width] ;
% area_new = [pointnew_11;pointnew_12;pointnew_21;pointnew_22] ;
% area_new = [100,100;280,100;100,160;280,160] ;
% Tran = calc_homography(area_old,area_new) ;
% Tran = maketform('projective',Tran);   %投影矩阵
% [imgn, X, Y] = imtransform(img,Tran);     %投影
% figure
% imshow(imgn) 
% hold on
% plot(col_min,col_min_row_min,'r*')
% plot(col_max,col_max_row_max,'r*')


%% get plate and draw plate in source image
%%% 在原图中绘制出车牌区域。
[rimg, ~] = drawPlate(img, save_con) ;
subplot(2,2,3)
imshow(rimg) % draw plate in origin image

%% get one plate to test algorithm
%%% 取出车牌进行预处理，滤波等
plate_temp = eximg{1} ; % only process first plate as example
gray_img = rgb2gray(plate_temp) ;
hcore = fspecial('gaussian') ;
gray_img = imfilter(gray_img, hcore) ;
t = graythresh(gray_img) ;
bin = im2bw(gray_img, t) ;
core = ones(2,2) ;
erode_bin = imerode(bin, core) ;
figure(5)
imshow(erode_bin)

%% get chars in plate
%%% 车牌字符分割
con = bwboundaries(erode_bin,8, 'noholes') ;
con_size = size(con) ;
con_size = con_size(1,1) ;
save_char = {} ;
inner = 1 ;
for i = 1:con_size
    dy = max(con{i}(:,1))-min(con{i}(:,1)) ;
    dx = max(con{i}(:,2)-min(con{i}(:,2))) ;
    if dy/dx <= char_wh_upper && dy/dx >= char_wh_lower && dx > 5 && dy > 10
        save_char{inner} = con{i} ;
        inner = inner+1 ;
    end
end % get char into a cell

%% re-get chars
%%% 根据先验知识分割车牌字符
exchar = regetChar(erode_bin, save_char) ;
exchar = imgNormal(exchar, char_nor_width,char_nor_height) ; % normalize the size of char image
size_char = size(exchar) ;
size_char = size_char(1,1) ;
charset = cell(size_char, 1) ;

%% thining the chars
%%% 细化车牌字符，使得其不会收到一些噪声干扰，前提是有一个较为稳健的特征提取方法。
% for i = 1:size_char
%     figure(3)
%     exchar{i} = uint8(exchar{i}*255) ;
%     charset{i} = bwmorph(exchar{i}, 'thin',Inf); % char thining
%     subplot(1, size_char, i)
%     imshow(exchar{i})
% end 
% 细化笔画
%%%%%%%%%%%%%%%%%%%up to now, get char, then need to recognize it%%%%%%%%%%


%% extract the char features and then classify them
%%% i could use projection to encode features, but it is not
%%% scale-invariant
%%% 提取字符特征，训练分类器识别或者采用模版匹配。

% test_path = 'G:\数据分析集合\plate\cha&num\' ;
% test_proj = zeros(35, char_nor_height+char_nor_width) ;
% for i = 1:34
%     name = [test_path, num2str(i),'.bmp'] ;
%     test_img = imread(name) ;
%     test_img_thin = bwmorph(test_img, 'thin',Inf); % char thining
%     proj1 = vertical_projection(test_img) ;
%     proj2 = horizonal_projection(test_img) ;
%     test_proj(i, :) = [proj1; proj2]' ;
% end
% save test_proj test_proj

load test_proj.mat
relation = [] ;
charpre_list = {} ;
for i = 2:7
    proj1 = vertical_projection(exchar{i}) ;
    proj2 = horizonal_projection(exchar{i}) ;
    proj = [proj1; proj2]' ;
    for j = 1:34
        tmp = corrcoef(proj,test_proj(j,:)) ;
        relation(j) = tmp(1,2) ;
    end
    [~, index] = max(relation) ;
    charpre_list{i-1} = getCharName(index) ;
end

for i =1:size_char
    figure(3)
    subplot(1, size_char, i)
    imshow(exchar{i})
    if i >= 2
        title(charpre_list{i-1})
    end
end


%% test  part
% % figure
% 
% % figure
% % subplot(1,2,1)
% l = 5 ;
% proj1 = vertical_projection(exchar{l}) ;
% proj2 = horizonal_projection(exchar{l}) ;
% a = [proj1', proj2'] ;
% a = mapminmax(a, 0, 1) ;
% % stem(a)
% % 
% % 
% % subplot(1,2,2)
% 
% relation = [] ;
% for i = 1:35
%     path = ['C:\Users\Administrator\Desktop\sample\cha&num\', num2str(i),'.bmp'];
% %     path = ['F:\opencvjpg\training sets\normalized\numbers\9\', 'n9_',num2str(i-1),'.bmp'] ;
%     img = imread(path) ;
%     img = img/255 ;
%     img = bwmorph(img, 'thin',Inf); % char thining
%     % subplot(1,2,2)
%     proj1 = vertical_projection(img) ;
%     proj2 = horizonal_projection(img) ;
%     b = [proj1', proj2'] ;
% %     stem([proj1', proj2'])
%     b = mapminmax(b, 0, 1) ;
%     tmp = corrcoef(a,b) ;
%     relation(i) = tmp(1,2) ;
% end
% % stem(relation)




