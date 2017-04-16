function merge_img = getBluePlate(img)
hsv_img = rgb2hsv(img) ;
h_img = hsv_img(:,:,1) ;
s_img = hsv_img(:,:,2) ;
v_img = hsv_img(:,:,3) ;
size_img = size(h_img) ;
h_tmp = zeros(size_img) ;
s_tmp = zeros(size_img) ;
v_tmp = zeros(size_img) ;
h_tmp((h_img >= 0.55 & h_img <= 0.72)) = 1 ;
s_tmp(s_img >= 0.4 & s_img < 1) = 1 ;
v_tmp(v_img >= 0.3 & v_img <= 1) = 1 ;
merge_img = h_tmp & s_tmp  & v_tmp;