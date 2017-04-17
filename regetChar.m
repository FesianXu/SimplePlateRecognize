function [charset, refine_charset] = regetChar(plate, char_con)
least_width = 10;
least_height = 10 ;
max_width = 40 ;
max_height = 40 ;
chars_set = {} ;
inner_loop = 1 ;
%% 删除太小和太大的字符假设区域
pre_charsize = length(char_con);
for i =1:pre_charsize
    row_max = max(char_con{i}(:,1)) ;
    row_min = min(char_con{i}(:,1)) ;
    col_max = max(char_con{i}(:,2)) ;
    col_min = min(char_con{i}(:,2)) ;
    drow = row_max-row_min ;
    dcol = col_max-col_min ;
    if (drow < least_height && dcol < least_width) || drow > max_height || dcol > max_width
    else
        chars_set{inner_loop} = char_con{i} ;
        inner_loop = inner_loop+1 ;
    end
end 

%% relocalize the center position of char
[loc_cell,len_chars_judged] = relocateChars(chars_set) ;
center_set = loc_cell{1} ; % 字符中心点

%% get all char according to prior knowledge
plate_size = size(plate) ;
plate_width = plate_size(2) ;
plate_height = plate_size(1) ;
plate_propo = zeros(len_chars_judged,1) ;
for i = 1:len_chars_judged
    plate_propo(i) = center_set(i, 2)/plate_width ;
end
type_list = decideCharType(plate_propo) ; % 知道了每个字符的位置，这个方法不太好，因为是对于车牌的，而不是相对位置的

%% 不管怎么样，但如果有一个字符的多个部分重复的时候，需要重合这部分，防止出现问题
%%% 合并chars_set和type_list
chars_set_combined = {} ;
inner_loop = 1 ;
for i = 1:7
    loc = find(type_list == i) ;
    tmpchars = [] ;
    for j = 1:length(loc)
        tmpchars = [tmpchars; chars_set{loc(j)}];
    end
    if isempty(tmpchars)
        continue ;
    end
    chars_set_combined{inner_loop} = tmpchars ;
    inner_loop = inner_loop+1 ;
end
for i = 1:7
    loc = find(type_list == i) ;
    if length(loc) > 1
        for j = 1:length(loc)-1
            type_list(loc(j)) = [] ;
        end
    end
end

[loc_cell,~] = relocateChars(chars_set_combined) ;
center_set = loc_cell{1} ; % 字符中心点
row_med = loc_cell{2} ;
col_med = loc_cell{3} ;
center_row_med = loc_cell{4} ;

%% get rest chars
all_type = 1:7 ;
dist = [uint16(0.1795*plate_width), uint16(0.1295*plate_width)];
diff_type = setdiff(all_type, type_list) ;
refine_charset = zeros(7,2) ; % 重新矫正之后的字符中心点集合
refine_charset(:, 1) = center_row_med ;
for i = 1:length(type_list)
    loc = find(type_list == type_list(i)) ;
    refine_charset(type_list(i), 2) = center_set(loc(1),2) ;
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
row_bias = 0 ;
col_bias = 0 ; % 小心越界,没有加越界保护
for i =1:7
    col_max = uint16(refine_charset(i,2)+col_med/2) ;
    col_min = uint16(refine_charset(i,2)-col_med/2) ;
    if col_min <= 0
        col_min = 1 ; 
    end
%     if col_min == 0
%         col_min = 1 ;
%     end
    charset{i} = plate(row_min-row_bias:row_max+row_bias,col_min-col_bias:col_max+col_bias) ;
end



