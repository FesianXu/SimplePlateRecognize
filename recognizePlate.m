%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: FesianXu
% ������ʶ���װ��һ������
% plate_cell ����Ԫ��,��ȡ�����ĳ���ͼ����Ҫ��׼����
% plate_img ���Ƶ�ԭͼ��
% chars_center �����ַ�������λ��
% correct_type ������������
% plate_type ����ʶ��֮����ַ�
% score ÿ���ַ�ʶ������ţ��÷֣�confidence
% img_color ԭͼƬ
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
% version1.5: radon transform first to judge whether the plate is slanted
% or not so that judge whether should correct the plate.
% version1.6: add svm to judge whether the plate area is a real plate or
% not
% version1.7: add line connection to refine the plate bw image and add
% rotate method to correct plate
%%%
function [plate_cell, plate_img,chars_center ,correct_type, plate_type, score] = recognizePlate(img_color)
tic ; % ��ʱ��ʼ
%% parameter initiation
%%% һЩ�����ĳ�ʼ�����趨
char_nor_width = 16*2 ;
char_nor_height = 32*2 ;
plate_nor_width = 160 ;
plate_nor_height = 48 ;
plate_wh_upper = 6 ; % ���Ƴ�������Ͻ�
plate_wh_lower = 1 ; % ���Ƴ�����½�
plate_wh_atlease_width = 50 ; % �������ٳ���
plate_wh_atlease_height = 30 ; % �������ٿ��
plate_area_norm_width = 300 ; % ���Ƽ����򳤶�
plate_area_norm_height = 140 ; % ���Ƽ�������
char_wh_upper = 2.5 ; % �ַ�������Ͻ�
char_wh_lower = 1.5 ; % �ַ�������½�
plate_area_width_bias = 10 ; % ���Ƽ������򳤶�ƫ��
plate_area_height_bias = 10 ; % ���Ƽ���������ƫ��
img_resize_width = 800 ;
img_resize_height = 600 ;
remove_error_plate_ratio = 0.7 ; % ���ٴ����Ƽ�����
slant_threshold = 4; % ����ĳ��������б��ֵ����Ҫ������
rotate_thresh = 8 ;
%%% output parameters
plate_cell = cell(7,1) ;
plate_type = zeros(7,1) ;
correct_type = 0 ;
score = zeros(7,1) ;
%%% �ȹ�һ����һ���еȵĳ߶ȣ����ұ������ܵĳ�������
%%% Ȼ���ڽϴ�߶�����ȡ�������������һ������
model_name = '.\train_data\isplate_svm.mat' ;
decide_chartype_svm_path = '.\train_data\decide_chartype_svm_model.mat' ;
numbers_svm_path = '.\train_data\numbers_svm_model.mat' ;
alphabet_svm_path = '.\train_data\alphabet_svm_model.mat' ;
model = load(model_name);
svm_model = model.svm_model ;
decide_chartype_svm = load(decide_chartype_svm_path) ;
decide_chartype_svm = decide_chartype_svm.Mdl ;
numbers_svm = load(numbers_svm_path) ;
numbers_svm = numbers_svm.Mdl ;
alphabet_svm = load(alphabet_svm_path) ;
alphabet_svm = alphabet_svm.Mdl ;

%% get image and turn it to hsv space
%%% ��������֪ʶ����ȡ������������Ҫע����ǣ���Ҫ�ų�һЩ���Եķǳ�����
%%% ��Ϊ�Ƕ�ͼƬ�������Ƕ���Ƶ�����Բ���Ҫ����̬ģ������
img_color_resize = imresize(img_color,[img_resize_height,img_resize_width]) ;
img_merge = getBluePlate(img_color_resize) ; %%% get blue area and mask blue area with pixel 1
erode_core = ones(1,1) ;
img_merge = imerode(img_merge, erode_core) ;
dilate_core = ones(10,20) ;
img_merge = imdilate(img_merge, dilate_core) ;
%%% ������̬ѧ�ղ������ó���������Ŀ���ֵͼ
img_merge_con = bwboundaries(img_merge,8, 'noholes') ; %% 8-��ͨ����
len_img_merge_con = length(img_merge_con) ;

%% search for plate location may be need to judge if it is really plate,delete some area
%%% ɾ����һЩ���Բ��ǳ��Ƽ�������ģ���ʱ���ƿ��ܻ���ڷ���任����͸�ӱ任����Ҫ��һ��
%%% �������
inner_loop = 1 ;
save_con = {} ; % ���澭�����Ƴߴ��Ų�ĳ��Ƽ�����
for i = 1:len_img_merge_con
    drow = max(img_merge_con{i}(:,1))-min(img_merge_con{i}(:,1)) ;
    dcol = max(img_merge_con{i}(:,2)-min(img_merge_con{i}(:,2))) ;
    if dcol/drow < plate_wh_upper && dcol/drow > plate_wh_lower && dcol > plate_wh_atlease_width && drow > plate_wh_atlease_height
        save_con{inner_loop} = img_merge_con{i} ;
        inner_loop = inner_loop+1 ;
    end
end

%% here we need to project transform to refine the plates
%%% ��ȡ�����Ƽ���������ĸ��߽ǣ�ӳ�䵽�������У���͸�ӱ任����������б��
%%% ���ýǶ�������

if isempty(save_con)
    disp('save con is empty!') ;
    return ;
end
[pl_img, ~] = extractPlate(img_color_resize, save_con) ;
pl_norm_img = cell(length(pl_img),1) ;
for i =1:length(pl_img)
    pl_norm_img{i} = imresize(pl_img{i}, [plate_area_norm_height, plate_area_norm_width]) ;
end %% ��׼�����Ƽ�����

%% ͨ���ų����̫С�����򣬼��ٳ��ƴ��������
total = zeros(length(pl_norm_img),1) ;
for i = 1:length(pl_norm_img)
    merimg = getBluePlate(pl_norm_img{i}) ;
    total(i) = bwarea(merimg) ;
end
max_total = max(total) ;
pl_img_tmp = {} ;
inner_loop = 1 ;
for i = 1:length(total)
    if total(i) >= max_total*remove_error_plate_ratio
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

%% ȡ��������Ƽ������ٸ����Ƿ��ܼ����ַ����ж��Ƿ��������ĳ���
%%% 
pl_norm_img_number = 1 ;
pl_judged_imgset = {} ; % �жϺ�ĳ�����
chars_judged_set = {} ; % �жϺ���ַ���
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
    %%% ��������д
    if isempty(imgt_save_con)
        disp('imgt save con is empty');
        return ;
    end

    %% radon�任�ж��Ƿ���Ҫ��������
    Icanny = edge(imgt_merge,'canny') ;
    theta = 1:180;
    [R,~] = radon(Icanny,theta);
    [~,J] = find(R >= max(max(R)));
    slant_angle = 90-J ;
    if abs(slant_angle) <= slant_threshold
        %%% ����Ҫ���� ��Ҫ��׼��
        imgn = imresize(img_test,[plate_nor_height,plate_nor_width]) ;
        correct_type = 1 ;
    elseif abs(slant_angle) > slant_threshold && abs(slant_angle) <= rotate_thresh
        imgn = imrotate(img_test,slant_angle,'bilinear','crop') ;
        correct_type = 2 ;
        %%%��б�ǶȲ���̫�󣬿���ͨ�����Ƶķ���任������
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
       %% ����ǵ�������������
        %%% ��б�Ƕ�̫���Ѿ������÷���任�����ˣ�����Ҫ����͸�ӱ任�����Ǵ���һ�������ر��εķ��ա�
        %%% TODO Ӧ�ø��ݳ����Ƿ���б�����о����Ƿ��ȡ���ƽ�����
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
        Tran = maketform('projective',Tran);   %ͶӰ����
        [imgn, X, Y] = imtransform(img_test,Tran);     %ͶӰ
        correct_type = 3 ;
    end

    %% ��������
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
    
    %% ɾ�����Ʊ߿�
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

    %% svm�������ж��Ƿ��ǳ���
    imgn_out = imresize(imgn_out, [plate_nor_height, plate_nor_width]) ;
    imvec = double(reshape(imgn_out,1,plate_nor_height*plate_nor_width)) ;
    [label, score] = predict(svm_model, imvec) ;
    if label == -1
        pl_norm_img_number = pl_norm_img_number+1 ;
        %%% �ó��Ƽ�����û�м�⵽�����ַ�����˻���һ�����Ƽ�����
    else
        %%% ���Ƽ��������г��ƣ��������һ�����ƺͳ����ַ�����
        pl_judged_imgset{pl_num_loop} = imgn_out ;
        pl_num_loop = pl_num_loop+1 ;
        pl_norm_img_number = pl_norm_img_number+1 ;
    end
end

%% re-get chars
%%% ��������֪ʶ�ָ���ַ�
%%% ��Ҫ��һ�������ָ���㷨
if isempty(pl_judged_imgset)
    disp('no license plate in this image!')
    return ;
end
%%% ȡ��һ�����ƽ��зָ��ַ�
judged_plate_num = 1 ;
imgn = pl_judged_imgset{judged_plate_num} ;
plate_img = imgn ;

imgn_con = bwboundaries(plate_img,8, 'noholes') ;
char_con = {} ;
inner = 1 ;
for i = 1:length(imgn_con)
    if length(imgn_con{i}(:,1)) < 20
        continue ;
    end
    drow = max(imgn_con{i}(:,1))-min(imgn_con{i}(:,1)) ;
    dcol = max(imgn_con{i}(:,2))-min(imgn_con{i}(:,2)) ;
    pval = drow/dcol ;
    if pval > 1.5 && pval < 5 && drow > 10 && dcol > 5 && drow < 40 && dcol < 20
        char_con{inner} = imgn_con{i} ; 
        inner = inner+1 ;
    end
end
%% segment the chars in plate
[exchar,center_set] = regetChar(imgn_out, char_con) ;
exchar = imgNormal(exchar, char_nor_width,char_nor_height) ; % normalize the size of char image
% figure
% for i = 1:length(exchar)
%     subplot(1,length(exchar),i)
%     imshow(exchar{i})
% end

for i = 1:length(exchar)
    plate_cell{i} = exchar{i} ;
end
chars_center = center_set ;
%% recognize plate chars
for i = 1:length(exchar)
    img = exchar{i} ;
    verproj = vertical_projection(img)' ;
    horproj = horizonal_projection(img)' ;
    allproj = reshape(img,1,length(img(:,1))*length(img(1,:))) ;
    proj = [verproj,horproj,allproj] ;
    [type_label,~] = predict(decide_chartype_svm,proj) ;
    if type_label == 1
        [plate_type(i),~] = predict(numbers_svm,proj) ;
    elseif type_label == 2
        [plate_type(i),~] = predict(alphabet_svm,proj) ;
    end
end

toc ; % ��ʱ����



