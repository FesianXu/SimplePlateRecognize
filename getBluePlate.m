function merge_img = getBluePlate(img)
hsv_img = rgb2hsv(img) ;
h_img = hsv_img(:,:,1) ;
s_img = hsv_img(:,:,2) ;
v_img = hsv_img(:,:,3) ;
tmp = h_img ;
tmp2 = s_img ;
tmp3 = tmp ;
tmp((h_img >= 0.60 & h_img <= 0.70)) = 1 ;
tmp((h_img < 0.56 | h_img > 0.71)) = 0 ;
tmp2(s_img >= 0.4 & s_img < 1) = 1 ;
tmp2(s_img < 0.4 | s_img > 1) = 0 ;
tmp3(v_img >= 0.3 & v_img <= 1) = 1 ;
tmp3(v_img < 0.3 | v_img > 1) = 0 ;
merge_img = tmp & tmp2 & tmp3 ;