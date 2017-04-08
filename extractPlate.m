function [eximg, num] = extractPlate(img, con_cell)
con_size = size(con_cell) ;
con_size = con_size(1,2) ;
eximg = cell(con_size, 1) ;
bias = 20 ;
for i = 1:con_size
    row_max = max(con_cell{i}(:,1));
    row_min = min(con_cell{i}(:,1));
    col_max = max(con_cell{i}(:,2)) ;
    col_min = min(con_cell{i}(:,2)) ;
    tmp_img = img(row_min:row_max+bias, col_min:col_max+bias, :) ;
    eximg{i} = tmp_img ;
end
num = con_size ;