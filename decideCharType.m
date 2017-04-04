function type_list = decideCharType(plate_pro_list)
char_list_size = size(plate_pro_list) ;
char_list_length = char_list_size(1,1) ; % number of chars
divide_points = [0.086, 0.216, 0.395, 0.525, 0.654, 0.784, 0.913] ; % 依靠着车牌字符的顺序排列的比例，左方向为初始方向
type_list = zeros(char_list_length, 1) ; % char's tpyes list
diff_list = zeros(length(divide_points),1) ; % difference list, select the smallest one
for i = 1:char_list_length
    for j = 1:length(divide_points)
        diff_list(j) = abs(plate_pro_list(i)-divide_points(j)) ;
    end
    [listmin,index] = min(diff_list) ;
    type_list(i) = index ;
end