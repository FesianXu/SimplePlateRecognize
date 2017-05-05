function [proj] = horizonal_projection(img)
[row col] = size(img) ;
proj = zeros(row,1);
for i = 1:row
    proj(i) = sum(img(i, :)) ;
end