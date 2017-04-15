function [points,left,right] = getPlateCorner(list)
%% 得到车牌角点
%%% points => [row,col]
%%% 有时候会出现点定位错的问题，那是因为前一个和后一个合在一起了，聚类出错，这个问题
%%% 有待解决，车牌需要标准化地大一点。
cosine_threshold_down = -0.7 ;
cosine_threshold_up = 0.7 ;
calc_step = 50 ; % 最好是偶数
sample_step = 1 ; % 采样步长
list_cos = [list;list(1:calc_step,:)] ;
list_cos = list_cos(1:sample_step:end,:) ;
lenlist = length(list_cos(:,1)) ;
tmpcos = zeros(lenlist-calc_step,1) ;
for i = 1:lenlist-calc_step
    p1 = list_cos(i,:) ;
    p2 = list_cos(i+calc_step,:) ;
    pmedian = list_cos(i+calc_step/2,:) ;
    p1 = [p1(1,2),p1(1,1)] ;
    p2 = [p2(1,2), p2(1,1)] ;
    pmedian = [pmedian(1,2), pmedian(1,1)] ;
    v1 = p1-pmedian ;
    v2 = p2-pmedian ;
    tmpcos(i) = v1*v2'/(norm(v1,2)*norm(v2,2)) ;
end
corner_list = [] ;
inner_loop = 1 ;
for i = 1:lenlist-calc_step
    if tmpcos(i) > cosine_threshold_down && tmpcos(i) < cosine_threshold_up
        corner_list(inner_loop,:) = list_cos(i+calc_step/2,:) ;
        inner_loop = inner_loop+1 ;
    end
end
max_col = max(list(:,2)) ;
min_col = min(list(:,1)) ;
bias = 50 ;
left = [] ;
right = [] ;
inner_loop = 1 ;
for i = 1:length(corner_list(:,1))
    if abs(corner_list(i,2)-min_col) <= bias
       left(inner_loop,:) = corner_list(i,:) ;
       inner_loop = inner_loop+1 ;
    end
end 
inner_loop = 1 ;
for i = 1:length(corner_list(:,1))
    if abs(corner_list(i,2)-max_col) <= bias
       right(inner_loop,:) = corner_list(i,:) ;
       inner_loop = inner_loop+1 ;
    end
end  % get left and right part
%% kmeans to cluster
[idx_left,~] = kmeans(left,2) ;
c1 = left(idx_left == 1,:) ;
c2 = left(idx_left == 2,:) ;
[idx_right,~] = kmeans(right,2) ;
c3 = right(idx_right == 1,:) ;
c4 = right(idx_right == 2,:) ;
p1 = c1(uint16(length(c1(:,1))/2),:) ;
p2 = c2(uint16(length(c2(:,1))/2),:) ;
p3 = c3(uint16(length(c3(:,1))/2),:) ;
p4 = c4(uint16(length(c4(:,1))/2),:) ;
if p1(1,1) < p2(1,1)
    p11 = p1 ;
    p21 = p2 ;
else
    p11 = p2 ;
    p21 = p1 ;
end
if p3(1,1) < p4(1,1)
    p12 = p3 ;
    p22 = p4 ;
else
    p12 = p4 ;
    p22 = p3 ;
end
points = [p11;p12;p21;p22] ;









