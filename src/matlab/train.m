%% training relative files
clc
clear 
close all

%% 判断是否是车牌的svm二分类器
% pos_path = 'F:\opencvjpg\training sets\save_raw_plate\' ;
% neg_path = 'F:\opencvjpg\training sets\save_abnormal_plate\';
% save_data = 'G:\电子工程\大学电子设计比赛项目\PlateRecognize\train_data\' ;
% 
% pos_dir = dir(pos_path);
% neg_dir = dir(neg_path) ;
% 
% vec_pos = zeros(length(pos_dir)-2,48*160) ;
% vec_neg = zeros(length(neg_dir)-2,48*160) ;
% for i = 1:length(pos_dir)
%     if pos_dir(i).isdir == 0
%         imgpos = imread([pos_path,pos_dir(i).name]) ;
%         imgpos = im2bw(imgpos,0.5) ;
%         imvec = reshape(imgpos,1,48*160) ;
%         vec_pos(i-2,:) = imvec ;
%     end
% end
% 
% for i = 1:length(neg_dir)
%     if neg_dir(i).isdir == 0
%         imgneg = imread([neg_path,neg_dir(i).name]) ;
%         imgneg = im2bw(imgneg,0.5) ;
%         imvec = reshape(imgneg,1,48*160) ;
%         vec_neg(i-2,:) = imvec ;
%     end
% end
% labels = zeros(length(pos_dir)+length(neg_dir)-4,1) ;
% labels(1:length(pos_dir)-2) = 1 ;
% labels(length(pos_dir)-1:end) = -1 ;
% traindata = [vec_pos; vec_neg];
% svm_model = fitcsvm(traindata, labels) ;
% model_name = 'G:\电子工程\大学电子设计比赛项目\PlateRecognize\train_data\isplate_svm' ;
% save(model_name,'svm_model')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% model_name = 'G:\电子工程\大学电子设计比赛项目\PlateRecognize\train_data\isplate_svm' ;
% model = load([model_name,'.mat']);
% svm_model = model.svm_model ;
% test_path = 'F:\opencvjpg\training sets\test_samples\' ;
% test_dir = dir(test_path) ;
% testlabels = [] ;
% posname = {} ;
% negname = {} ;
% posloop = 1;
% negloop = 1 ;
% for i = 3:length(test_dir)
%     imgtest = imread([test_path,test_dir(i).name]) ;
%     imgtest = im2bw(imgtest,0.5) ;
%     imvec = double(reshape(imgtest,1,48*160)) ;
%     [label, score] = predict(svm_model, imvec) ;
%     testlabels(i-2) = label ;
%     if label == 1
%         posname{posloop} = test_dir(i).name ;
%         posloop = posloop+1 ;
%     else
%         negname{negloop} = test_dir(i).name ;
%         negloop = negloop+1 ;
%     end
% end














%% 车牌字符svm
% 简单的特征，垂直投影+水平投影+全体像素

set_path = '..\..\res\trainning_set\normalized\' ;
alp_path = [set_path, 'alphabet\'] ;
numbers_path = [set_path, 'numbers\'] ;
neg_path = [set_path, 'neg\'] ;
feature_mat_path = '..\..\train_data\feature.mat' ;
label_mat_path = '..\..\train_data\label.mat' ;
svm_model_path = '..\..\train_data\char_svm_model.mat' ;
decide_chartype_svm_path = '..\..\train_data\decide_chartype_svm_model.mat' ;
type_mat_path = '..\..\train_data\type.mat' ;
numbers_svm_path = '..\..\train_data\numbers_svm_model.mat' ;
alphabet_svm_path = '..\..\train_data\alphabet_svm_model.mat' ;

alp_dir = dir(alp_path) ;
numbers_dir = dir(numbers_path) ;
neg_dir = dir(neg_path) ;

feature_mat = [] ; % [feature], feature matrix
labels_mat = [] ; % decide the char labels
type_mat = [] ; % decide which type of the char, including numbers,alphabet

inner_loop = 1 ;
tic ;

%% 数据提取
% for i = 1:length(numbers_dir)
%     if strcmp(numbers_dir(i).name, '.') == 0 && strcmp(numbers_dir(i).name, '..') == 0
%         imgf_path = [numbers_path,numbers_dir(i).name,'\'] ;
%         inner_dir = dir(imgf_path) ;
%         for j = 1:length(inner_dir)
%             if inner_dir(j).isdir == 0
%                 img_path = [imgf_path, inner_dir(j).name] ;
%                 img = imread(img_path) ;
%                 verproj = vertical_projection(img)' ;
%                 horproj = horizonal_projection(img)' ;
%                 allproj = reshape(img,1,length(img(:,1))*length(img(1,:))) ;
%                 proj = [verproj,horproj,allproj] ;
%                 feature_mat(inner_loop,:) = proj ;
%                 labels_mat(inner_loop) = i+22 ;
%                 type_mat(inner_loop) = 1 ;
%                 inner_loop = inner_loop+1 ;
%             end
%         end
%     end
% end
% 
% number2alp_divide = length(feature_mat(:,1)) ;
% for i = 1:length(alp_dir)
%     if strcmp(alp_dir(i).name, '.') == 0 && strcmp(alp_dir(i).name, '..') == 0
%         imgf_path = [alp_path,alp_dir(i).name,'\'] ;
%         inner_dir = dir(imgf_path) ;
%         for j = 1:length(inner_dir)
%             if inner_dir(j).isdir == 0
%                 img_path = [imgf_path, inner_dir(j).name] ;
%                 img = imread(img_path) ;
%                 verproj = vertical_projection(img)' ;
%                 horproj = horizonal_projection(img)' ;
%                 allproj = reshape(img,1,length(img(:,1))*length(img(1,:))) ;
%                 proj = [verproj,horproj,allproj] ;
%                 feature_mat(inner_loop,:) = proj ;
%                 labels_mat(inner_loop) = i-2 ;
%                 type_mat(inner_loop) = 2 ;
%                 inner_loop = inner_loop+1 ;
%             end
%         end
%     end
% end
% alp2neg_divide = length(feature_mat(:,1)) ;
% save(feature_mat_path, 'feature_mat') ;
% save(label_mat_path, 'labels_mat') ;
% save(type_mat_path, 'type_mat') ;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 训练分类器，判断是否是数字还是字母
% feature = struct2cell(load(feature_mat_path)) ;
% feature = feature{1} ;
% type = struct2cell(load(type_mat_path)) ;
% type = type{1} ;
t = templateSVM('Standardize',1) ;
Mdl = fitcecoc(feature,type,'Learners',t);
% save(decide_chartype_svm_path,'Mdl') ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 分别训练数字和字母的svm
% num2alp_point = 0 ;
% feature = struct2cell(load(feature_mat_path)) ;
% feature = feature{1} ;
% labels = struct2cell(load(label_mat_path)) ;
% labels = labels{1} ;
% for i = 1:length(labels(1,:))
%     if labels(i) < 25 || labels(i) > 34
%         num2alp_point = i ;
%         break ;
%     end
% end
% num2alp_point = num2alp_point-1 ;
% fture_1 = feature(1:num2alp_point,:) ;
% fture_2 = feature(num2alp_point+1:end,:) ;
% labels_1 = labels(1:num2alp_point) ;
% labels_2 = labels(num2alp_point+1:end) ;
% t1 = templateSVM('Standardize',1) ;
% Mdl = fitcecoc(fture_1,labels_1,'Learners',t1);
% save(numbers_svm_path,'Mdl') ;
% t2 = templateSVM('Standardize',1) ;
% Mdl = fitcecoc(fture_2,labels_2,'Learners',t2);
% save(alphabet_svm_path,'Mdl') ;

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% feature = struct2cell(load(feature_mat_path)) ;
% feature = feature{1} ;
% labels = struct2cell(load(label_mat_path)) ;
% labels = labels{1} ;
% t = templateSVM('Standardize',1) ;
% Mdl = fitcecoc(feature,labels,'Learners',t);
% save(svm_model_path,'Mdl') ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% svmmodel = load(svm_model_path) ;
% svmmodel = svmmodel.Mdl ;
% CVMdl = crossval(svmmodel);
% oosLoss = kfoldLoss(CVMdl)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% imgdir = dir('G:\电子工程\大学电子设计比赛项目\PlateRecognize\res\test_set\bonus\') ;
% for i = 3:4
%     img = imread(['G:\电子工程\大学电子设计比赛项目\PlateRecognize\res\test_set\bonus\',imgdir(i).name]) ;
%     img = im2bw(img,0.5) ;
%     verproj = vertical_projection(img)' ;
%     horproj = horizonal_projection(img)' ;
%     allproj = reshape(img,1,length(img(:,1))*length(img(1,:))) ;
%     proj = [verproj,horproj,allproj] ;
%     [labels,score] = predict(svmmodel,proj) ;
% 
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% only numbers
% 0.0229 both v and h and all
% 0.0243 all
% 0.1051 ver
% 0.0539 hor

% numbers+alphabets
% 0.0734 with 507.6 seconds

toc ;





