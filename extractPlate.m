function [eximg, num] = extractPlate(img, con_cell)
con_size = size(con_cell) ;
con_size = con_size(1,2) ;
eximg = cell(con_size, 1) ;
bias = 0 ;
for i = 1:con_size
    x_max = max(con_cell{i}(:,1));
    x_min = min(con_cell{i}(:,1));
    y_max = max(con_cell{i}(:,2)) ;
    y_min = min(con_cell{i}(:,2)) ;
    tmp_img = img(x_min-bias:x_max+bias, y_min-bias:y_max+bias, :) ;
    eximg{i} = tmp_img ;
end
num = con_size ;