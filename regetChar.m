function [charset] = regetChar(plate, char_con)
pre_charsize = size(char_con) ;
pre_charsize = pre_charsize(1,2) ;
for i =1:pre_charsize
    row_max = max(char_con{i}(:,1)) ;
    row_min = min(char_con{i}(:,1)) ;
    col_max = max(char_con{i}(:,2)) ;
    col_min = min(char_con{i}(:,2)) ;
    dx = row_max-row_min ;
    dy = col_max-col_min ;
    if dx < 10 && dy < 10
        char_con(i) = [] ;
    end
end % 删除太小的字符假设区域
%% relocalize the center position of char
dxdy_set = zeros(pre_charsize, 2) ;
center_set = zeros(pre_charsize, 2) ;
for i =1:pre_charsize
    row_max = max(char_con{i}(:,1)) ;
    row_min = min(char_con{i}(:,1)) ;
    col_max = max(char_con{i}(:,2)) ;
    col_min = min(char_con{i}(:,2)) ;
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
plate_propo = zeros(pre_charsize,1) ;
for i = 1:pre_charsize
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



