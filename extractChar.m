function [exchar, num] = extractChar(plate, char_cell)
 size_char = size(char_cell) ;
 size_char = size_char(1,2) ;
 exchar = cell(size_char, 1) ;
 num = size_char ;
 exchar = {} ;
 bias = 0 ;
 for i = 1:size_char
    row_max = max(char_cell{i}(:,1)) ;
    row_min = min(char_cell{i}(:,1)) ;
    col_max = max(char_cell{i}(:,2)) ;
    col_min = min(char_cell{i}(:,2)) ;
    tmp_img = plate(row_min:row_max, col_min-bias:col_max+bias) ;
    exchar{i} = tmp_img ;
 end