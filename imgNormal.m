function [nor_img] = imgNormal(exchar, width, height)
len_cell = length(exchar) ;
nor_img = cell(len_cell,1) ;
for i = 1:len_cell
    resize_img = imresize(exchar{i}, [height, width]) ;
    nor_img{i} = resize_img ;
end
