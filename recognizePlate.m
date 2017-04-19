%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: FesianXu
% ������ʶ���װ��һ������
% plate_cell ����Ԫ��,��ȡ�����ĳ���ͼ����Ҫ��׼����
% plate_img ���Ƶ�ԭͼ��
% chars_center �����ַ�������λ��
% plate_type ����ʶ��֮����ַ�
% score ÿ���ַ�ʶ������ţ��÷֣�confidence
% img_color ԭͼƬ
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [plate_cell, plate_img,chars_center ,plate_type, score] = recognizePlate(img_color)
%% parameter initiation
%%% һЩ�����ĳ�ʼ�����趨
char_nor_width = 16*2 ;
char_nor_height = 32*2 ;
plate_nor_width = 160 ;
plate_nor_height = 48 ;
plate_wh_upper = 6 ; % ���Ƴ�������Ͻ�
plate_wh_lower = 1 ; % ���Ƴ�����½�
plate_wh_atlease_width = 30 ; % �������ٳ���
plate_wh_atlease_height = 30 ; % �������ٿ��
plate_area_norm_width = 300 ; % ���Ƽ����򳤶�
plate_area_norm_height = 140 ; % ���Ƽ�������
char_wh_upper = 5 ; % �ַ�������Ͻ�
char_wh_lower = 1 ; % �ַ�������½�
plate_area_width_bias = 10 ; % ���Ƽ������򳤶�ƫ��
plate_area_height_bias = 10 ; % ���Ƽ���������ƫ��
img_resize_width = 800 ;
img_resize_height = 600 ;
plate_cell = cell(7,1) ;
plate_type = zeros(7,1) ;
score = zeros(7,1) ;
%%% �ȹ�һ����һ���еȵĳ߶ȣ����ұ������ܵĳ�������
%%% Ȼ���ڽϴ�߶�����ȡ�������������һ������
tic ; % ��ʱ��ʼ
%% get image and turn it to hsv space
%%% ��������֪ʶ����ȡ������������Ҫע����ǣ���Ҫ�ų�һЩ���Եķǳ�����
%%% ��Ϊ�Ƕ�ͼƬ�������Ƕ���Ƶ�����Բ���Ҫ����̬ģ������
img_color_resize = imresize(img_color,[img_resize_height,img_resize_width]) ;
img_merge = getBluePlate(img_color_resize) ; %%% get blue area and mask blue area with pixel 1
erode_core = ones(2,2) ;
img_merge = imerode(img_merge, erode_core) ;
dilate_core = ones(10,10) ;
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
    %%% ��������д
    if isempty(imgt_save_con)
        disp('imgt save con is empty');
        return ;
    end

    %% ����ǵ�������������
    %%% ��������
    %%% TODO Ӧ�ø��ݳ����Ƿ���б�����о����Ƿ��ȡ���ƽ�����
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
    Tran = maketform('projective',Tran);   %ͶӰ����
    [imgn, X, Y] = imtransform(img_test,Tran);     %ͶӰ

    %% ��������
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


    %% ɾ�����Ʊ߿�

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
    %%% �����ַ��ָ�
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
        %%% �ó��Ƽ�����û�м�⵽�����ַ�����˻���һ�����Ƽ�����
    else
        %%% ���Ƽ��������г��ƣ��������һ�����ƺͳ����ַ�����
        pl_judged_imgset{pl_num_loop} = imgn_out ;
        chars_judged_set{pl_num_loop} = save_chars_con ;
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
%%% �õ����ڼ��������µ�ģ��ƥ�䣬����������֮�󣬿��Բ���������ȡ+��������ķ���
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
toc ; % ��ʱ����



