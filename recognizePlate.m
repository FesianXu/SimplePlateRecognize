%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: FesianXu
% 将车牌识别封装成一个函数
% plate_cell 车牌元胞,提取出来的车牌图像，需要标准化？
% plate_img 车牌的原图像
% chars_center 车牌字符的中心位置
% plate_type 经过识别之后的字符
% score 每个字符识别的自信，得分，confidence
% img_color 原图片
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [plate_cell, plate_img,chars_center ,plate_type, score] = recognizePlate(img_color)
%% parameter initiation
%%% 一些参数的初始化和设定
char_nor_width = 16*2 ;
char_nor_height = 32*2 ;
plate_nor_width = 160 ;
plate_nor_height = 48 ;
plate_wh_upper = 6 ; % 车牌长宽比例上界
plate_wh_lower = 1 ; % 车牌长宽比下界
plate_wh_atlease_width = 30 ; % 车牌至少长度
plate_wh_atlease_height = 30 ; % 车牌至少宽度
plate_area_norm_width = 300 ; % 车牌假设域长度
plate_area_norm_height = 140 ; % 车牌假设域宽度
char_wh_upper = 5 ; % 字符长宽比上界
char_wh_lower = 1 ; % 字符长宽比下界
plate_area_width_bias = 10 ; % 车牌假设区域长度偏移
plate_area_height_bias = 10 ; % 车牌假设区域宽度偏移
img_resize_width = 800 ;
img_resize_height = 600 ;
plate_cell = cell(7,1) ;
plate_type = zeros(7,1) ;
score = zeros(7,1) ;
%%% 先归一化到一个中等的尺度，并且遍历可能的车牌区域
%%% 然后在较大尺度中提取出车牌域进行下一步处理。
tic ; % 计时开始
%% get image and turn it to hsv space
%%% 根据先验知识，提取出车牌区域，需要注意的是，需要排除一些明显的非车牌域
%%% 因为是读图片，而不是读视频，所以不需要做动态模糊处理。
img_color_resize = imresize(img_color,[img_resize_height,img_resize_width]) ;
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

%%
%%% test part , address multiple plate problem
% figure(20)
% for i = 1:length(pl_norm_img)
%     testimg = pl_norm_img{i} ;
%     subplot(1,length(pl_norm_img),i)
%     imshow(testimg)
% end
%%%%

%% 取出多个车牌假设域，再根据是否能检测出字符而判断是否是真正的车牌
%%% 
pl_norm_img_number = 1 ;
pl_judged_imgset = {} ; % 判断后的车牌域
chars_judged_set = {} ; % 判断后的字符域
pl_num_loop = 1 ;
while pl_norm_img_number <= length(pl_norm_img)
    img_test = pl_norm_img{pl_norm_img_number} ;
    imgt_merge = getBluePlate(img_test) ;
    img_dilate_core = ones(10,20) ;
    imgt_merge = imdilate(imgt_merge, img_dilate_core) ;
    imgt_con = bwboundaries(imgt_merge,8, 'noholes') ;
    len_imgt_con = length(imgt_con) ;
    inner_loop = 1 ;
    imgt_save_con = {} ;
    for i = 1:len_imgt_con
        drow = max(imgt_con{i}(:,1))-min(imgt_con{i}(:,1)) ;
        dcol = max(imgt_con{i}(:,2))-min(imgt_con{i}(:,2)) ;
%         dcol/drow
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
    %%% TODO 应该根据车牌是否倾斜而自行决定是否采取车牌矫正。
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
%     figure(3)
%     subplot(2,1,1)
%     imshow(img_test)
%     hold on
%     plot(p11(1,1),p11(1,2),'g*')
%     plot(p12(1,1),p12(1,2),'r*')
%     plot(p21(1,1),p21(1,2),'b*')
%     plot(p22(1,1),p22(1,2),'y*')
%     % hold on
%     plot(left(:,2),left(:,1),'r*')
%     plot(right(:,2),right(:,1),'b*')
%     subplot(2,1,2)
%     imshow(imgn)
%     figure(4)
%     imshow(imgt_merge)


    %% 删除车牌边框

    imgn_merge = getBluePlate(imgn) ;
    img_dilate_core = ones(5,5) ;
    imgn_merge = imdilate(imgn_merge, img_dilate_core) ;
    imgn_con = bwboundaries(imgn_merge,8, 'noholes') ;
    [imgn_set, ~] = extractPlate(imgn, imgn_con) ;
    if isempty(imgn_set)
        disp('imgn_set is empty!')
        return ;
    end
    imgn = imgn_set{1} ;

    imgn_gray = rgb2gray(imgn) ;
    bw_thres = graythresh(imgn_gray) ;
    imgn_bw = im2bw(imgn_gray,bw_thres) ;
    imgn_out = deletePlateFrame(imgn_bw) ;
    erode_core = ones(2,2) ;
    imgn_out = imerode(imgn_out, erode_core) ;
    imgn_out = im2bw(imgn_out,0.6) ;
%     figure(10)
%     imshow(imgn_out)

    %% get chars in plate
    %%% 车牌字符分割
    chars_con = bwboundaries(imgn_out,8, 'noholes') ;
    len_chars_con = length(chars_con) ;
    inner_loop = 1 ;
    save_chars_con = {} ;
    for i = 1:len_chars_con
        drow = max(chars_con{i}(:,1))-min(chars_con{i}(:,1)) ;
        dcol = max(chars_con{i}(:,2)-min(chars_con{i}(:,2))) ;
        if drow/dcol <= char_wh_upper && drow/dcol >= char_wh_lower && dcol > 5 && drow > 10
            save_chars_con{inner_loop} = chars_con{i} ;
            inner_loop = inner_loop+1 ;
        end
    end

%     figure(10)
%     hold on
%     for i = 1:length(save_chars_con)
%         plot(save_chars_con{i}(:,2),save_chars_con{i}(:,1),'r*')
%     end

    if isempty(save_chars_con)
        pl_norm_img_number = pl_norm_img_number+1 ;
        %%% 该车牌假设域没有检测到车牌字符，因此换下一个车牌假设域
    else
        %%% 车牌假设域内有车牌，保存其归一化车牌和车牌字符链表
        pl_judged_imgset{pl_num_loop} = imgn_out ;
        chars_judged_set{pl_num_loop} = save_chars_con ;
        pl_num_loop = pl_num_loop+1 ;
        pl_norm_img_number = pl_norm_img_number+1 ;
    end
end

%% re-get chars
%%% 根据先验知识分割车牌字符
%%% 需要进一步调整分割的算法
if isempty(pl_judged_imgset)
    disp('no license plate in this image!')
    return ;
end
%%% 取第一个车牌进行分割字符
judged_plate_num = 1 ;
imgn_out = pl_judged_imgset{judged_plate_num} ;
chars_con = chars_judged_set{judged_plate_num} ;

% figure
% imshow(imgn_out)

% for i = 1:length(chars_con)
%     hold on
%     plot(chars_con{i}(:,2),chars_con{i}(:,1),'r*')
% end
[exchar,center_set] = regetChar(imgn_out, chars_con) ;
exchar = imgNormal(exchar, char_nor_width,char_nor_height) ; % normalize the size of char image
% plot(center_set(:,2), center_set(:,1), 'b*')
% figure
% for i = 1:7
%     subplot(1,7,i) ;
%     imshow(exchar{i})
% end

%% recognize the chars in plate
%%% 用的是在极少样本下的模版匹配，当样本多了之后，可以采用特征提取+多分类器的方法
% load test_proj.mat
% relation = [] ;
% charpre_list = {} ;
% for i = 2:7
%     proj1 = vertical_projection(exchar{i}) ;
%     proj2 = horizonal_projection(exchar{i}) ;
%     proj = [proj1; proj2]' ;
%     for j = 1:34
%         tmp = corrcoef(proj,test_proj(j,:)) ;
%         relation(j) = tmp(1,2) ;
%     end
%     [~, index] = max(relation) ;
%     charpre_list{i-1} = getCharName(index) ;
% end

% for i =1:length(exchar)
%     figure(3)
%     subplot(1, length(exchar), i)
%     imshow(exchar{i})
% %     if i >= 2
% %         title(charpre_list{i-1})
% %     end
% end
for i = 1:length(exchar)
    plate_cell{i} = exchar{i} ;
end
plate_img = imgn_out ;
chars_center = center_set ;
toc ; % 计时结束



