function [loc_cell, len_chars_judged] = relocateChars(chars_set)
%% 重定位字符中心
%%% loc_cell [center_set, row_med, col_med, center_row_med]
%%% len_chars_judged 判别出的字符数量
%%% 不同字符的中心点， 理想字符宽度， 理想字符长度， 理想中心位置
loc_cell = {} ;
len_chars_judged = length(chars_set) ;
dxdy_set = zeros(len_chars_judged, 2) ;
center_set = zeros(len_chars_judged, 2) ;
for i =1:len_chars_judged
    row_max = max(chars_set{i}(:,1)) ;
    row_min = min(chars_set{i}(:,1)) ;
    col_max = max(chars_set{i}(:,2)) ;
    col_min = min(chars_set{i}(:,2)) ;
    dxdy_set(i,1) = row_max-row_min ;
    dxdy_set(i,2) = col_max-col_min ;
    center_set(i,1) = uint8((row_max+row_min)*0.5) ;
    center_set(i,2) = uint8((col_max+col_min)*0.5) ;
end
row_med = median(dxdy_set(:,1)) ;
col_med = median(dxdy_set(:,2)) ; % ideal char width and height
center_row_med = median(center_set(:,1)) ;
center_set(:,1) = center_row_med ; % ideal center position but not complete
loc_cell{1} = center_set ;
loc_cell{2} = row_med ;
loc_cell{3} = col_med ;
loc_cell{4} = center_row_med ;
end

