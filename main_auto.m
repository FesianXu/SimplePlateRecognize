%%%%
% Author: FesianXu
% ȫ�Զ���ȡͼƬ������recognizePlate�⺯�����Լ���д��
%%%%
clc
clear all
close all
%% get a frame of image
path = 'F:\opencvjpg\' ;
file_name = '1109.jpg' ; 
save_path = 'F:\opencvjpg\training sets\save_raw_char\' ;
save_plate_path = 'F:\opencvjpg\training sets\save_raw_plate\' ;
%%% 1014 1016 1026 big problem, 71 is too gray 1071 addressed !
%%% 1043 34 71 multiple test addressed!
%%% 1080 1120 regetchar failed, the cell have been over 8 list addressed!
%%% 1016 imgNormal failed
%%% 74 refine failed
%%% 1014 ���ֶ�λ����
%%% �ڳ��Ʒֱ��ʱȽϵ͵�ʱ����ܻ����һЩС���� 49
%%% a little problem in chars segment in 1110
%%% 1016 �ָ�Ч�������ر��
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
invalid_file_name = {} ;
for j = 3:length(dirimg)
    if dirimg(j).isdir == 1
        continue ;
    end
    fprintf('��%d��ͼƬ\r\n',j-2) ;
    file_name = [path,dirimg(j).name] ;
    img_color = imread(file_name) ;
    try
        [plate_cell, plate_img, chars_center,plate_type, score] = recognizePlate(img_color) ;
    catch
        invalid = invalid+1 ;
        invalid_file_name{invalid} = dirimg(j).name ;
        continue ;
    end
    if isempty(plate_cell{1})
        invalid = invalid+1 ;
        invalid_file_name{invalid} = dirimg(j).name ;
        continue ;
    end
    %%% save plate img into folder
    save_plate_name = [save_plate_path, num2str(j-2),'---',dirimg(j).name] ;
%     imwrite(plate_img, save_plate_name) ;
    imshow(plate_img)
    hold on
    plot(chars_center(:,2),chars_center(:,1),'r*')
    valid = valid+1 ;
    for i = 1:length(plate_cell)
        save_name = [save_path, num2str(j-2),'--',num2str(i),'.jpg'];
        imwrite(plate_cell{i},save_name) ;
    end
end
fprintf('��ЧͼƬ��%d\r\n',valid) ;
fprintf('��ЧͼƬ��%d\r\n',invalid) ;
fprintf('ͼƬ����%d\r\n',valid+invalid);
save_mat_name = [date,'-',num2str(now),'-invalid_file_name.mat'];
save_mat_path = 'G:\���ӹ���\��ѧ������Ʊ�����Ŀ\PlateRecognize\test_record\' ;
save_mat_file_path = [save_mat_path, save_mat_name] ;
save(save_mat_file_path, 'invalid_file_name')


