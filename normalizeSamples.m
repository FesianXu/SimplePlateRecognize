%%%
% 标准化样本
%%%
clc
clear
close all

rawset_path = '.\res\trainning_set\raw\' ;
norset_path = '.\res\trainning_set\normalized\' ;
nor_width = 32 ;
nor_heigth = 64 ;
isclear = 'no' ;
isnormalize = 'yes' ;

alp_raw = dir([rawset_path,'alphabet\']) ;
kanji_raw = dir([rawset_path,'kanji\']) ;
neg_raw = dir([rawset_path,'neg\']) ;
number_raw = dir([rawset_path,'numbers\']) ;
inner_loop = 1;

%% clear 
if strcmp(isclear,'yes') == 1
    alp_nor = dir([norset_path,'alphabet\']) ;
    for i = 1:length(alp_nor)
        if strcmp(alp_nor(i).name,'.') == 0 && strcmp(alp_nor(i).name,'..') == 0
            imgf_path = [norset_path,'alphabet\',alp_nor(i).name,'\'] ;
            img_dir = dir(imgf_path) ;
            for j = 1:length(img_dir)
                if img_dir(j).isdir == 0
                    file_name = [imgf_path, img_dir(j).name] ;
                    delete(file_name) ;
                end
            end
        end
    end
    number_nor = dir([norset_path,'numbers\']) ;
    for i = 1:length(number_nor)
        if strcmp(number_nor(i).name,'.') == 0 && strcmp(number_nor(i).name,'..') == 0
            imgf_path = [norset_path,'numbers\',number_nor(i).name,'\'] ;
            img_dir = dir(imgf_path) ;
            for j = 1:length(img_dir)
                if img_dir(j).isdir == 0
                    file_name = [imgf_path, img_dir(j).name] ;
                    delete(file_name) ;
                end
            end
        end
    end
    neg_nor = dir([norset_path,'neg\']) ;
    for i = 1:length(neg_nor)
        if neg_nor(i).isdir == 0
            file_name = [norset_path,'neg\',neg_nor(i).name] ;
            delete(file_name) ;
        end
    end
end


tic ;
if strcmp(isnormalize,'yes') == 1
    %% normalize alphabet
    for i = 1:length(alp_raw)
        if strcmp(alp_raw(i).name,'.') == 0 && strcmp(alp_raw(i).name,'..') == 0
            imgf_path = [rawset_path,'alphabet\',alp_raw(i).name,'\'] ;
            inner_dir = dir(imgf_path) ;
            for j = 1:length(inner_dir)
                if inner_dir(j).isdir == 0
                    img = imread([imgf_path, inner_dir(j).name]) ;
                    img = imresize(img,[nor_heigth, nor_width]) ;
                    img = im2bw(img, 0.5) ;
                    save_path = [norset_path, 'alphabet\',alp_raw(i).name,'\'] ;
                    save_name = [save_path, alp_raw(i).name,'-',num2str(inner_loop),'.png'] ;
                    imwrite(img, save_name) ;
                    inner_loop = inner_loop+1 ;
                end
            end
        end
    end
    %% normalize numbers
    inner_loop = 1 ;
    for i = 1:length(number_raw)
        if strcmp(number_raw(i).name,'.') == 0 && strcmp(number_raw(i).name,'..') == 0
            imgf_path = [rawset_path,'numbers\',number_raw(i).name,'\'] ;
            inner_dir = dir(imgf_path) ;
            for j = 1:length(inner_dir)
                if inner_dir(j).isdir == 0
                    img = imread([imgf_path, inner_dir(j).name]) ;
                    img = imresize(img,[nor_heigth, nor_width]) ;
                    img = im2bw(img, 0.5) ;
                    save_path = [norset_path, 'numbers\',number_raw(i).name,'\'] ;
                    save_name = [save_path, number_raw(i).name,'-',num2str(inner_loop),'.png'] ;
                    imwrite(img, save_name) ;
                    inner_loop = inner_loop+1 ;
                end
            end
        end
    end

    %% normalize neg
    inner_loop = 1 ;
    for i = 1:length(neg_raw)
        if neg_raw(i).isdir ~= 1
            imgf_path = [rawset_path,'neg\',neg_raw(i).name] ;
            img = imread(imgf_path) ;
            img = imresize(img,[nor_heigth, nor_width]) ;
            img = im2bw(img, 0.5) ;
            save_path = [norset_path, 'neg\'] ;
            save_name = [save_path, 'neg-',num2str(inner_loop),'.png'] ;
            imwrite(img, save_name) ;
            inner_loop = inner_loop+1 ;
        end
    end
end
toc ;









