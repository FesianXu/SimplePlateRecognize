%%%%
% Author: FesianXu
% ȫ�Զ���ȡͼƬ������recognizePlate�⺯�����Լ���д��
%%%%
clc
clear
close all
%% get a frame of image
path = 'F:\opencvjpg\test_img\' ;
file_name = '1109.jpg' ; 
save_path = 'F:\opencvjpg\training sets\save_raw_char\' ;
save_plate_path = 'F:\opencvjpg\training sets\save_raw_plate\' ;
auto_reg_save_path = '..\..\res\auto_samples_set\' ;
run_platform = 'MATLAB R2016a' ;

dirimg = dir(path) ;
invalid = 0 ;
valid = 0 ;
warning('off')
invalid_file_name = {} ;
cor_type_name = cell(3,1) ;
cor_counter = ones(3,1) ;

t_begin = cputime ;
for j = 3:length(dirimg)
    if dirimg(j).isdir == 1
        continue ;
    end
    fprintf('��%d��ͼƬ\r\n',j-2) ;
    file_name = [path,dirimg(j).name] ;
    img_color = imread(file_name) ;
    try
        [plate_cell, plate_img, chars_center,correct_type,plate_type, score] = recognizePlate(img_color) ;
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
    %% save plate img into folder
    plate_result = '' ;
    for i = 1:length(plate_type)
        plate_result = [plate_result, getCharName(plate_type(i))] ;
    end
    plate_result(1) = '+' ;
    save_plate_name = [save_plate_path, num2str(j-2),'-',plate_result,'-',dirimg(j).name] ;
    imwrite(plate_img, save_plate_name) ;
    %% save correction types and image's names
    if correct_type ~= 0
        cor_type_name{correct_type,cor_counter(correct_type)} = dirimg(j).name ;
        cor_counter(correct_type) = cor_counter(correct_type)+1 ;
    end
    valid = valid+1 ;
    %% save chars
    for i = 1:length(plate_cell)
        save_name = [save_path, num2str(j-2),'--',num2str(i),'.jpg'];
        imwrite(plate_cell{i},save_name) ;
    end
    %% begin to save auto recognize samples
    saveRecognizeChars(plate_cell, plate_type, auto_reg_save_path) ;

end
t_end = cputime ;
%%% run record logging
%% save record mat files
fprintf('��ЧͼƬ��%d\r\n',valid) ;
fprintf('��ЧͼƬ��%d\r\n',invalid) ;
fprintf('ͼƬ����%d\r\n',valid+invalid);
save_mat_name = [date,'-',num2str(now),'-invalid_file_name.mat'];
save_mat_path = '.\test_record\mat_files\' ;
save_mat_file_path = [save_mat_path, save_mat_name] ;
save(save_mat_file_path, 'invalid_file_name')

%% save record in txt files 
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
    fprintf(fid,'%s\r\n',invalid_file_name{i}) ;
end
fprintf(fid,'\r\n---------------------------------------------\r\n') ;
fprintf(fid,'��ЧͼƬ��%d\r\n',valid) ;
fprintf(fid,'��ЧͼƬ��%d\r\n',invalid) ;
fprintf(fid,'ͼƬ����%d\r\n',valid+invalid);
fprintf(fid,'��Ч��%.2f\r\n',valid/(valid+invalid));
fprintf(fid,'����ʱ��%f��\r\n',t_end-t_begin) ;
%% save correction types
for i = 1:length(cor_type_name(:,1))
    counter_inner = 0 ;
    fprintf(fid,'\r\n---------------------------------------------\r\n') ;
    fprintf(fid,'�ɵ�%d�ֽ��������������ļ���������ʼ\r\n',i) ;
    for j = 1:length(cor_type_name(i,:))
        if isempty(cor_type_name{i,j}) == 0
            fprintf(fid,'%s\r\n',cor_type_name{i,j});
            counter_inner = counter_inner+1 ;
        end
    end
    fprintf(fid,'�ɵ�%d�ֽ��������������ļ�,��%d��\r\n',i,counter_inner) ;
end
%% ��ע
fprintf(fid,'\r\n---------------------------------------------\r\n') ;
fprintf(fid,'��һ�ֽ����������ǳ�����б�Ƕȹ�С������Ҫ����\r\n') ;
fprintf(fid,'�ڶ��ֽ����������ǳ�����б�Ƕ����У����Խ��ƿ����Ƿ���任��������ת����\r\n') ;
fprintf(fid,'��һ�ֽ����������ǳ�����б�Ƕȹ���ͨ��͸�ӱ任����\r\n') ;
fprintf(fid,'���л���Ϊ��%s\r\n',run_platform) ;
fprintf(fid,'\r\n---------------------------------------------\r\n') ;
fclose(fid) ;

