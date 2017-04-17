function [eximg, num] = extractPlate(img, con_cell)
con_size = length(con_cell) ;
eximg = cell(con_size, 1) ;
bias = 0 ;
inner_loop = 1 ;
least_length = 100 ;
for i = 1:con_size
    if length(con_cell{i}) > least_length
        row_max = max(con_cell{i}(:,1));
        row_min = min(con_cell{i}(:,1));
        col_max = max(con_cell{i}(:,2)) ;
        col_min = min(con_cell{i}(:,2)) ;
        tmp_img = img(row_min-bias:row_max+bias, col_min-bias:col_max+bias, :) ;
        eximg{inner_loop} = tmp_img ;
        inner_loop = inner_loop+1 ;
    end %% 太小的不保存，不是车牌区域，是噪声点。
end
num = inner_loop ;