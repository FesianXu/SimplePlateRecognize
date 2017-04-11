%%%
% Author:FesianXu
% the teaching demo of plate detection
% version1.0: simple function, not affine transform, not machine learning
% algorithms,not smart char divide algorithms
% version1.1: add projective transform to refine the plate ,with simple
% recognization technique.
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
char_wh_upper = 4 ; % 字符长宽比上界
char_wh_lower = 1 ; % 字符长宽比下界
plate_area_width_bias = 10 ; % 车牌假设区域长度偏移
plate_area_height_bias = 10 ; % 车牌假设区域宽度偏移

%% get image and turn it to hsv space
%%% 进行一些预处理之后，根据先验知识初步提取出车牌假设区域。
path = 'F:\opencvjpg\' ;
file_name = '1018.jpg' ;
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
    if dx/dy < plate_wh_upper && dx/dy > plate_wh_lower && dx > 20 && dy > 20
        save_con{inner} = con{i} ;
        inner = inner+1 ;
    end
end

%% here we need to project transform to refine the plates
%%% 提取出车牌假设区域的四个边角，映射到新区域中，做透视变换，矫正其倾斜。
%%% 连续求导自迭代法
% [eximg, ~] = extractPlate(img, save_con) ; % extract plate area
% eximg = imgNormal(eximg, plate_nor_width, plate_nor_height) ;

[pl_img, ~] = extractPlate(img, save_con) ;
img_t = pl_img{1} ; 
merge = getBluePlate(img_t) ;
merge = imdilate(merge, core) ;
con_n = bwboundaries(merge,8, 'noholes') ;
con_n_size = size(con_n) ;
con_n_size = con_n_size(1,1) ;
save_n_con = {} ;
inner = 1 ;
for i = 1:con_n_size
    dy = max(con_n{i}(:,1))-min(con_n{i}(:,1)) ;
    dx = max(con_n{i}(:,2))-min(con_n{i}(:,2)) ;
    if dx/dy < plate_wh_upper && dx/dy > plate_wh_lower && dx > 20 && dy > 20
        save_n_con{inner} = con_n{i} ;
        inner = inner+1 ;
    end
end
list = save_n_con{1} ;

drow = max(list(:,1))-min(list(:,1)) ;
dcol = max(list(:,2))-min(list(:,2)) ;
prop = dcol/drow 
if prop <= 2.5
    col_gap = 30 ;
    row_step = 30 ;
    delta = [] ;
    list_new = [list; list(1:row_step,:)] ; % 循环后缀
    list_len = length(list(:,1)) ;
    for i = 1:list_len
        delta(i) = list_new(i+row_step,1)-list_new(i,1) ;
    end
    delta_len = length(delta) ;
    del2 = [] ;
    for i = 1:delta_len-1
        del2(i) = delta(i+1)-delta(i) ;
    end
    max_delta = max(delta) ;
    delta_m = abs(delta) >= max_delta-1 ;
    del2_m = del2 == 0 ;
    del2_m(length(delta_m)) = 0; 
    delta_m = [0,delta_m] ;
    del2_m = [0,del2_m] ;
    tina = delta_m.*del2_m ;
    tina_len = length(tina) ;
    tii = [] ;
    for i = 1:tina_len-1
        tii(i) = tina(i+1)-tina(i) ;
    end
    % figure
    % plot(delta)
    [~,loc1] = find(tii == 1) ;
    [~,loc2] = find(tii == -1) ;
    figure
    subplot(2,1,1)
    imshow(img_t)
    hold on
    loc_len = length(loc1) ;
%     for i = 1:loc_len
%         plot(list_new(loc1(i),2), list_new(loc1(i),1),'r*') ;
%         plot(list_new(loc2(i)+row_step,2),list_new(loc2(i)+row_step,1),'b*')
%     end
    subplot(2,1,2)
    imshow(merge)
%     hold on
%     loc_len = length(loc1) ;
%     for i = 1:loc_len
%         plot(list_new(loc1(i),2), list_new(loc1(i),1),'r*') ;
%         plot(list_new(loc2(i)+row_step,2),list_new(loc2(i)+row_step,1),'b*')
%     end
    [~,loc_max_row] = max(list(:,1)) ;
    [~,loc_min_row] = min(list(:,1)) ;
    [~,loc_max_col] = max(list(:,2)) ;
    [~,loc_min_col] = min(list(:,2)) ;
    max_row = list(loc_max_row,:) ;
    min_row = list(loc_min_row,:) ;
    max_col = list(loc_max_col,:) ;
    min_col = list(loc_min_col,:) ;
    subplot(2,1,1)
    hold on
%     plot(max_row(:,2),max_row(:,1),'g*')
%     plot(min_row(:,2),min_row(:,1),'g*')
%     plot(max_col(:,2),max_col(:,1),'g*')
%     plot(min_col(:,2),min_col(:,1),'g*')
%     subplot(2,1,2)
%     hold on
%     plot(max_row(:,2),max_row(:,1),'g*')
%     plot(min_row(:,2),min_row(:,1),'g*')
%     plot(max_col(:,2),max_col(:,1),'g*')
%     plot(min_col(:,2),min_col(:,1),'g*')
    
    point1 = [min_col(:,2),min_col(:,1)] ;
    point2 = [min_row(:,2),min_row(:,1)] ;
    point3 = [max_row(:,2),max_row(:,1)];
    point4 = [max_col(:,2),max_col(:,1)] ;
%     area_old = [point2;point4;point1;point3] ;
    
    
    poss_area = [point1;point2;point3;point4] ; % 列排序优先
    for i = 1:loc_len
        poss_area(i+4,:) = [list_new(loc1(i),2), list_new(loc1(i),1)] ;
    end
    len = length(poss_area(:,1)) ;
    for i = 1:loc_len
        poss_area(len+i,:) = [list_new(loc2(i)+row_step,2),list_new(loc2(i)+row_step,1)] ;
    end
    [idx,c] = kmeans(poss_area,2) ;
    c1 = poss_area(idx == 1,:) ;
    c2 = poss_area(idx == 2,:) ;
    ctmp = {c1,c2} ;
    if c1(1,1) > c2(1,1)
        c1 = ctmp{2} ;
        c2 = ctmp{1} ;
    end
    [~,c1_max_loc] = max(c1(:,2)) ;
    [~,c1_min_loc] = min(c1(:,2)) ;
    [~,c2_max_loc] = max(c2(:,2)) ;
    [~,c2_min_loc] = min(c2(:,2)) ;
    p1 = c1(c1_min_loc,:);
    p2 = c2(c2_min_loc,:) ;
    p3 = c1(c1_max_loc,:) ;
    p4 = c2(c2_max_loc,:) ;
    area_old = [p1;p2;p3;p4] ;
    figure(2)
    subplot(2,1,2)
    hold on
    plot(p1(1,1),p1(1,2),'r*')
    plot(p2(1,1),p2(1,2),'b*')
    plot(p3(1,1),p3(1,2),'g*')
    plot(p4(1,1),p4(1,2),'k*')
    subplot(2,1,1)
    for i = 1:length(poss_area(:,1))
        plot(poss_area(i,1),poss_area(i,2),'r*')
    end
    area_new = [100,100;100+plate_nor_width,100;100,100+plate_nor_height;100+plate_nor_width,100+plate_nor_height] ;
    a = 1 ;

    Tran = calc_homography(area_old,area_new) ;
    Tran = maketform('projective',Tran);   %投影矩阵
    [imgn, X, Y] = imtransform(img_t,Tran);     %投影
    figure(10)
    subplot(2,1,1)
    imshow(img_t)
    subplot(2,1,2)
    imshow(imgn) 
    
    
else
    [eximg, ~] = extractPlate(img, save_con) ; % extract plate area
    eximg = imgNormal(eximg, plate_nor_width, plate_nor_height) ;
    imgn = eximg{1} ;
end
% 
% figure(10)
% subplot(3,1,1)
% stem(delta)
% subplot(3,1,2)
% stem(del2)
% subplot(3,1,3)
% stem(tii)
%%% 可以优化，多重求导图的平滑处理



%% get plate and draw plate in source image
%%% 在原图中绘制出车牌区域。
% [rimg, ~] = drawPlate(img, save_con) ;
% subplot(2,2,3)
% imshow(rimg) % draw plate in origin image

%% get one plate to test algorithm
%%% 取出车牌进行预处理，滤波等
plate_temp = imgn ; % only process first plate as example
gray_img = rgb2gray(plate_temp) ;
hcore = fspecial('gaussian') ;
gray_img = imfilter(gray_img, hcore) ;
% t = graythresh(gray_img) ;
bin = im2bw(gray_img, 0.5) ;
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
figure(5)
hold on
lenc = length(save_char) ;
for i = 1:lenc
    plot(save_char{i}(:,2),save_char{i}(:,1),'r*')
end


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

%% recognize the chars in plate

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




