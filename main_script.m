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
% version1.4: address multiple plate areas in an image
%%%

clc
clear
close all
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
char_wh_upper = 3.5 ; % 字符长宽比上界
char_wh_lower = 1.5 ; % 字符长宽比下界
plate_area_width_bias = 10 ; % 车牌假设区域长度偏移
plate_area_height_bias = 10 ; % 车牌假设区域宽度偏移
img_resize_width = 800 ;
img_resize_height = 600 ;
%%% 先归一化到一个中等的尺度，并且遍历可能的车牌区域
%%% 然后在较大尺度中提取出车牌域进行下一步处理。
model_name = '.\train_data\isplate_svm.mat' ;
model = load(model_name);
svm_model = model.svm_model ;

tic ; % 计时开始
%% get image and turn it to hsv space
%%% 根据先验知识，提取出车牌区域，需要注意的是，需要排除一些明显的非车牌域
%%% 因为是读图片，而不是读视频，所以不需要做动态模糊处理。
path = 'F:\opencvjpg\' ;
file_name = '19.jpg' ; 
%%% 1014 1016 1026 big problem, 71 is too gray 1071 addressed !
%%% 1043 34 71 multiple test addressed!
%%% 1080 1120 regetchar failed, the cell have been over 8 list addressed!
%%% 1016 imgNormal failed
%%% 74 refine failed
%%% 1014 汉字定位问题
%%% 在车牌分辨率比较低的时候可能会出现一些小问题 49
%%% a little problem in chars segment in 1110
%%% 1016 分割效果不是特别好
file_path = [path, file_name] ;
img_color = imread(file_path) ;
img_color_resize = imresize(img_color,[img_resize_height,img_resize_width]) ;
img_merge = getBluePlate(img_color_resize) ; %%% get blue area and mask blue area with pixel 1
erode_core = ones(1,1) ;
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

%% 通过排除面积太小的区域，减少车牌错误假设域
total = zeros(length(pl_norm_img),1) ;
for i = 1:length(pl_norm_img)
    merimg = getBluePlate(pl_norm_img{i}) ;
    total(i) = bwarea(merimg) ;
end
max_total = max(total) ;
pl_img_tmp = {} ;
inner_loop = 1 ;
for i = 1:length(total)
    if total(i) >= max_total*0.7
        pl_img_tmp(inner_loop) = pl_norm_img(i) ;
        inner_loop = inner_loop+1 ;
    end
end
pl_norm_img = pl_img_tmp ;

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
    img_dilate_core = ones(10,15) ;
    imgt_merge = imdilate(imgt_merge, img_dilate_core) ;
    imgt_con = bwboundaries(imgt_merge,8, 'noholes') ;
    len_imgt_con = length(imgt_con) ;
    inner_loop = 1 ;
    imgt_save_con = {} ;
    for i = 1:len_imgt_con
        drow = max(imgt_con{i}(:,1))-min(imgt_con{i}(:,1)) ;
        dcol = max(imgt_con{i}(:,2))-min(imgt_con{i}(:,2)) ;
        if dcol/drow < plate_wh_upper && dcol/drow > plate_wh_lower && dcol > plate_wh_atlease_width && drow > plate_wh_atlease_height
            imgt_save_con{inner_loop} = imgt_con{i} ;
            inner_loop = inner_loop+1 ;
        end
    end
    %%% 车牌域特写
    if isempty(imgt_save_con)
        disp('imgt save con is empty');
        return ;
    end

    %% radon变换判断是否需要矫正车牌
    Icanny = edge(imgt_merge,'canny') ;
    theta = 1:180;
    [R,~] = radon(Icanny,theta);
    [~,J] = find(R >= max(max(R)));
    slant_angle = 90-J ;
    if abs(slant_angle) <= 3
        %%% 不需要矫正 需要标准化
        imgn = imresize(img_test,[plate_nor_height,plate_nor_width]) ;
        correct_type = 1 ;
    elseif abs(slant_angle) > 3 && abs(slant_angle) <= 15
        imgn = imrotate(img_test,slant_angle,'bilinear','crop') ;
        correct_type = 2 ;
        %%%倾斜角度不算太大，可以通过近似的仿射变换矫正。
    else
        %% line refine
        step = 140 ;
        lenspace = 20;
        if length(imgt_save_con) ~= 1
            list_len = zeros(length(imgt_save_con),1) ;
            for i = 1:length(imgt_save_con)
                list_len(i) = length(imgt_save_con{i});
            end
            [~, idx_list] = max(list_len) ;
        else
            idx_list = 1 ;
        end
        con = imgt_save_con{idx_list} ;
        con = [con(:,:);con(1:step,:)] ;
        imgt_merge = uint8(imgt_merge)*255 ;
        for i = 1:lenspace:length(con)-step
            imgt_merge = insertShape((imgt_merge), 'Line',[con(i,2),con(i,1),con(i+step,2),con(i+step,1)], 'LineWidth',6, 'Color', 'white') ;
        end
        imgt_merge = im2bw(imgt_merge, 0.6);
        imgt_save_con = bwboundaries(imgt_merge,8, 'noholes') ;
       %% 捕获角点特征矫正车牌
        %%% 倾斜角度太大，已经不能用仿射变换近似了，所以要采用透视变换，但是存在一定更严重变形的风险。
        %%% TODO 应该根据车牌是否倾斜而自行决定是否采取车牌矫正。
        if length(imgt_save_con) ~= 1
            list_len = zeros(length(imgt_save_con),1) ;
            for i = 1:length(imgt_save_con)
                list_len(i) = length(imgt_save_con{i});
            end
            [~, idx_list] = max(list_len) ;
        else
            idx_list = 1 ;
        end
        list = imgt_save_con{idx_list} ;
%         k = convhull(list(:,1),list(:,2)) ;
%         list = list(k,:) ;
        [points,~,~] = getPlateCorner(list) ;
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
        correct_type = 3 ;
    end

    %% 矫正测试
%     figure(30)
%     subplot(2,1,1)
%     imshow(img_test)
%     hold on
%     plot(p11(1,1),p11(1,2),'g*')
%     plot(p12(1,1),p12(1,2),'r*')
%     plot(p21(1,1),p21(1,2),'b*')
%     plot(p22(1,1),p22(1,2),'y*')
%     % hold on
% %     plot(left(:,2),left(:,1),'r*')
% %     plot(right(:,2),right(:,1),'b*')
%     subplot(2,1,2)
%     imshow(imgn)
%     figure(4)
%     imshow(imgt_merge) 
    
    %% 删除车牌边框
    imgn_merge = getBluePlate(imgn) ;
    img_dilate_core = ones(5,15) ;
    imgn_merge = imdilate(imgn_merge, img_dilate_core) ;
    imgn_con = bwboundaries(imgn_merge,8, 'noholes') ;
    [imgn_set, ~] = extractPlate(imgn, imgn_con) ;
    if isempty(imgn_set)
        disp('imgn_set is empty!')
        return ;
    end
    list_len = zeros(length(imgn_set),1) ;
    for i = 1:length(imgn_set)
        list_len(i) = length(imgn_set{i});
    end
    [~, idx_list] = max(list_len) ;
    imgn = imgn_set{idx_list} ;

    imgn_gray = rgb2gray(imgn) ;
    bw_thres = graythresh(imgn_gray) ;
    imgn_bw = im2bw(imgn_gray,bw_thres) ;
    imgn_out = deletePlateFrame(imgn_bw) ;
    erode_core = ones(2,2) ;
    imgn_out = imerode(imgn_out, erode_core) ;
    imgn_out = im2bw(imgn_out,0.6) ;

    %% svm分类器判断是否是车牌
    imgn_out = imresize(imgn_out, [plate_nor_height, plate_nor_width]) ;
    imvec = double(reshape(imgn_out,1,plate_nor_height*plate_nor_width)) ;
    [label, score] = predict(svm_model, imvec) ;
    if label == -1
        pl_norm_img_number = pl_norm_img_number+1 ;
        %%% 该车牌假设域没有检测到车牌字符，因此换下一个车牌假设域
    else
        %%% 车牌假设域内有车牌，保存其归一化车牌和车牌字符链表
        pl_judged_imgset{pl_num_loop} = imgn_out ;
        pl_num_loop = pl_num_loop+1 ;
        pl_norm_img_number = pl_norm_img_number+1 ;
    end
end

plate_img = pl_judged_imgset{1} ;
figure
imshow(plate_img)
[charset] = getChars(plate_img) ;
figure(10)
for i = 1:length(charset)
    subplot(1,length(charset),i)
    imshow(charset{i})
end


toc ; % 计时结束

