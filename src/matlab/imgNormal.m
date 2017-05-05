function [nor_img] = imgNormal(exchar, width, height)
len_cell = length(exchar) ;
nor_img = cell(len_cell,1) ;
inner_loop = 1;
for i = 1:len_cell
    if isempty(exchar{i})
        continue ;
    end
    resize_img = imresize(exchar{i}, [height, width]) ;
    nor_img{inner_loop} = resize_img ;
    inner_loop = inner_loop+1 ;
end
