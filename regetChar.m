function [charset] = regetChar(plate, char_con)
least_width = 10;
least_height = 10 ;
max_width = 40 ;
max_height = 40 ;
chars_set = {} ;
inner_loop = 1 ;
%% 删除太小的字符假设区域
pre_charsize = length(char_con);
for i =1:pre_charsize
    row_max = max(char_con{i}(:,1)) ;
    row_min = min(char_con{i}(:,1)) ;
    col_max = max(char_con{i}(:,2)) ;
    col_min = min(char_con{i}(:,2)) ;
    drow = row_max-row_min ;
    dcol = col_max-col_min ;
    if (drow < least_height && dcol < least_width) || drow > 40 || dcol > 40
    else
        chars_set{inner_loop} = char_con{i} ;
        inner_loop = inner_loop+1 ;
    end
end 
%% relocalize the center position of char
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
%% get all char according to prior knowledge
plate_size = size(plate) ;
plate_width = plate_size(2) ;
plate_height = plate_size(1) ;
plate_propo = zeros(len_chars_judged,1) ;
for i = 1:len_chars_judged
    plate_propo(i) = center_set(i, 2)/plate_width ;
end
type_list = decideCharType(plate_propo) ; % 知道了每个字符的位置，这个方法不太好，因为是对于车牌的，而不是相对位置的
%%% get rest chars
all_type = 1:7 ;
dist = [uint16(0.1795*plate_width), uint16(0.1295*plate_width)];
diff_type = setdiff(all_type, type_list) ;
refine_charset = zeros(7,2) ; % 重新矫正之后的字符中心点集合
refine_charset(:, 1) = center_row_med ;
for i = 1:length(type_list)
    loc = find(type_list == type_list(i)) ;
    refine_charset(type_list(i), 2) = center_set(loc,2) ;
end
for i = 1:length(diff_type)
    min_diff = abs(type_list-diff_type(i)) ;
    [diff_v, ind] = min(min_diff) ;
    ind_type = type_list(ind) ;
    type_list_loc = find(type_list == ind_type) ;
    if ind_type > 2 % 引导点在分割点右侧
        if diff_type(i) > ind_type % 需要补全的字符在引导点右侧
            refine_charset(diff_type(i),2) = center_set(type_list_loc, 2)+diff_v*dist(2);
        else % 左侧
            if diff_type(i) <= 2
                refine_charset(diff_type(i),2) = center_set(type_list_loc, 2)-dist(1)-(diff_v-1)*dist(2);
            else
                refine_charset(diff_type(i),2) = center_set(type_list_loc, 2)-diff_v*dist(2);
            end
        end
    else % 在分割点左侧
        if diff_type(i) > ind_type % 需要补全的字符在引导点右侧
            if diff_type(i) > 2
                refine_charset(diff_type(i),2) = center_set(type_list_loc, 2)+dist(1)+(diff_v-1)*dist(2);
            else
                refine_charset(diff_type(i),2) = center_set(type_list_loc, 2)+diff_v*dist(2);
            end
        else % 左侧
            refine_charset(diff_type(i),2) = center_set(type_list_loc, 2)-diff_v*dist(2);
        end
    end
end
%% extrat char set
charset = {} ;
row_max = uint8(center_row_med+row_med/2) ;
row_min = uint8(center_row_med-row_med/2) ;
for i =1:7
    col_max = uint16(refine_charset(i,2)+col_med/2) ;
    col_min = uint16(refine_charset(i,2)-col_med/2) ;
    if col_min <= 0
        continue 
    end
    charset{i} = plate(row_min:row_max,col_min:col_max) ;
end



