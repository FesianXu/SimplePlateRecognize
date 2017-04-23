%% test file
clc
clear 
close all

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
model_name = 'G:\电子工程\大学电子设计比赛项目\PlateRecognize\train_data\isplate_svm' ;
model = load([model_name,'.mat']);
svm_model = model.svm_model ;
test_path = 'F:\opencvjpg\training sets\test_samples\' ;
test_dir = dir(test_path) ;
testlabels = [] ;
posname = {} ;
negname = {} ;
posloop = 1;
negloop = 1 ;
for i = 3:length(test_dir)
    imgtest = imread([test_path,test_dir(i).name]) ;
    imgtest = im2bw(imgtest,0.5) ;
    imvec = double(reshape(imgtest,1,48*160)) ;
    [label, score] = predict(svm_model, imvec) ;
    testlabels(i-2) = label ;
    if label == 1
        posname{posloop} = test_dir(i).name ;
        posloop = posloop+1 ;
    else
        negname{negloop} = test_dir(i).name ;
        negloop = negloop+1 ;
    end
end


