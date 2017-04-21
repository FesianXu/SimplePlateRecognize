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
t_begin = cputime ;
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
    imwrite(plate_img, save_plate_name) ;
%     imshow(plate_img)
%     hold on
%     plot(chars_center(:,2),chars_center(:,1),'r*')
    valid = valid+1 ;
    for i = 1:length(plate_cell)
        save_name = [save_path, num2str(j-2),'--',num2str(i),'.jpg'];
        imwrite(plate_cell{i},save_name) ;
    end
end
t_end = cputime ;
fprintf('��ЧͼƬ��%d\r\n',valid) ;
fprintf('��ЧͼƬ��%d\r\n',invalid) ;
fprintf('ͼƬ����%d\r\n',valid+invalid);
save_mat_name = [date,'-',num2str(now),'-invalid_file_name.mat'];
save_mat_path = '.\test_record\mat_files\' ;
save_mat_file_path = [save_mat_path, save_mat_name] ;
save(save_mat_file_path, 'invalid_file_name')
%%% save record mat files
save_txt_path = '.\test_record\txt_files\' ;
save_txt_name = [date,'-',num2str(now),'-invalid_file_name.txt'] ;
save_txt_file_path = [save_txt_path, save_txt_name] ;
fid = fopen(save_txt_file_path, 'wt') ;
len_invalid = length(invalid_file_name) ;
fprintf(fid,'%s\r\n',date) ;
fprintf(fid,'ͼƬ�ļ���:%s\r\n',path) ;
fprintf(fid,'\r---------------------------------------------\r') ;
fprintf(fid,'��ЧͼƬ�б�:\r\n') ;
for i = 1:len_invalid
    fprintf(fid,'%s\r',invalid_file_name{i}) ;
end
fprintf(fid,'\r\n---------------------------------------------\r\n') ;
fprintf(fid,'��ЧͼƬ��%d\r\n',valid) ;
fprintf(fid,'��ЧͼƬ��%d\r\n',invalid) ;
fprintf(fid,'ͼƬ����%d\r\n',valid+invalid);
fprintf(fid,'��Ч��%.2f\r\n',valid/(valid+invalid));
fprintf(fid,'����ʱ��%f��\r\n',t_end-t_begin) ;
fclose(fid) ;
%%% save record in txt files 





