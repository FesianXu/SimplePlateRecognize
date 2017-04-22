function merge_img = getBluePlate(img)
%% parameters init
h_img_up = 0.72 ;
h_img_down = 0.55 ;
s_img_up = 1.0 ;
s_img_down = 0.4 ;
v_img_up = 0.90 ;
v_img_down = 0.30 ;
%% select blue area, but need to develop in uneven illumination env
hsv_img = rgb2hsv(img) ;
h_img = hsv_img(:,:,1) ;
s_img = hsv_img(:,:,2) ;
v_img = hsv_img(:,:,3) ;
size_img = size(h_img) ;
h_tmp = zeros(size_img) ;
s_tmp = zeros(size_img) ;
v_tmp = zeros(size_img) ;
h_tmp((h_img >= h_img_down & h_img <= h_img_up)) = 1 ;
s_tmp(s_img >= s_img_down & s_img < s_img_up) = 1 ;
v_tmp(v_img >= v_img_down & v_img <= v_img_up) = 1 ;
merge_img = h_tmp & s_tmp  & v_tmp;
