%%%
% Author:FesianXu
% the teaching demo of plate detection
%%%
clc
clear
close all

%% parameter initiation
char_nor_width = 16 ;
char_nor_height = 32 ;
plate_nor_width = 160 ;
plate_nor_height = 48 ;

%% get image and turn it to hsv space
path = 'F:\\opencvjpg\\' ;
file_name = '17.jpg' ;
file_path = [path, file_name] ;
img = imread(file_path) ;
merge = getBluePlate(img) ;
% get hsv blue
core = ones(6,6) ;
merge = imdilate(merge, core) ;
% subplot(2,2,1)
% imshow(img)
% subplot(2,2,2)
% imshow(merge)
con = bwboundaries(merge,8, 'noholes') ;
con_size = size(con) ;
con_size = con_size(1,1) ;
save_con = {} ;
inner = 1 ;

%% search for plate location may be need to judge if it is really plate
for i = 1:con_size
    dy = max(con{i}(:,1))-min(con{i}(:,1)) ;
    dx = max(con{i}(:,2)-min(con{i}(:,2))) ;
    if dx/dy < 4 && dx/dy > 1.5 && dx > 20 && dy > 20
        save_con{inner} = con{i} ;
        inner = inner+1 ;
    end
end

%% get plate and draw plate in source image
[rimg num] = drawPlate(img, save_con) ;
% subplot(2,2,3)
% imshow(rimg) % draw plate in origin image
[eximg, num] = extractPlate(img, save_con) ;
eximg = imgNormal(eximg, plate_nor_width, plate_nor_height) ;

%% get one plate to test algorithm
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
con = bwboundaries(erode_bin,8, 'noholes') ;
con_size = size(con) ;
con_size = con_size(1,1) ;
save_char = {} ;
inner = 1 ;
for i = 1:con_size
    dy = max(con{i}(:,1))-min(con{i}(:,1)) ;
    dx = max(con{i}(:,2)-min(con{i}(:,2))) ;
    if dy/dx <= 2.5 && dy/dx >= 1.2 && dx > 5 && dy > 10
        save_char{inner} = con{i} ;
        inner = inner+1 ;
    end
end % get char into a cell

%% re-get chars
exchar = regetChar(erode_bin, save_char) ;
exchar = imgNormal(exchar, char_nor_width,char_nor_height) ; % normalize the size of char image
size_char = size(exchar) ;
size_char = size_char(1,1) ;
charset = cell(size_char, 1) ;
%% thining the chars
for i = 1:size_char
%     figure(3)
    exchar{i} = uint8(exchar{i}*255) ;
    charset{i} = bwmorph(exchar{i}, 'thin',Inf); % char thining
%     subplot(1, size_char, i)
%     imshow(exchar{i})
end 
% Ï¸»¯±Ê»­
%%%%%%%%%%%%%%%%%%%up to now, get char, then need to recognize it%%%%%%%%%%
%% extract the char features and then classify them
%%% i could use projection to encode features, but it is not
%%% scale-invariant

test_path = 'C:\Users\Administrator\Desktop\sample\cha&num\' ;
test_proj = zeros(35, char_nor_height+char_nor_width) ;
for i = 1:34
    name = [test_path, num2str(i),'.bmp'] ;
    test_img = imread(name) ;
    test_img_thin = bwmorph(test_img, 'thin',Inf); % char thining
    proj1 = vertical_projection(test_img) ;
    proj2 = horizonal_projection(test_img) ;
    test_proj(i, :) = [proj1; proj2]' ;
end
relation = [] ;
for i = 2:7
    proj1 = vertical_projection(exchar{i}) ;
    proj2 = horizonal_projection(exchar{i}) ;
    proj = [proj1; proj2]' ;
    for j = 1:34
        tmp = corrcoef(proj,test_proj(j,:)) ;
        relation(j) = tmp(1,2) ;
    end
    [~, index] = max(relation) ;
    getCharName(index) 
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




