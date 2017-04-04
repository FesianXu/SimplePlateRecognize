function [exchar, num] = extractChar(plate, char_cell)
 size_char = size(char_cell) ;
 size_char = size_char(1,2) ;
 exchar = cell(size_char, 1) ;
 num = size_char ;
 exchar = {} ;
 bias = 0 ;
 for i = 1:size_char
    x_max = max(char_cell{i}(:,1)) ;
    x_min = min(char_cell{i}(:,1)) ;
    y_max = max(char_cell{i}(:,2)) ;
    y_min = min(char_cell{i}(:,2)) ;
    tmp_img = plate(x_min:x_max, y_min-bias:y_max+bias) ;
    exchar{i} = tmp_img ;
 end