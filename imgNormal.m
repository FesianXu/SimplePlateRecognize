function [nor_img] = imgNormal(exchar, width, height)
size_cell = size(exchar) ;
size_cell = max(size_cell) ;
nor_img = cell(size_cell,1) ;
for i = 1:size_cell
    resize_img = imresize(exchar{i}, [height, width]) ;
    nor_img{i} = resize_img ;
end
