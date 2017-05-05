function [proj] = vertical_projection(img)
[row col] = size(img) ;
proj = zeros(col,1);
for i = 1:col
    proj(i) = sum(img(:, i)) ;
end