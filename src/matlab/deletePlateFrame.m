function img_out = deletePlateFrame(img)
%% ȥ�����ƿ�ܣ�ͨ��ˮƽ��ֵ�䶯�ж�
horizon_threshold = 10 ;
sizeimg = size(img) ;
img_out = zeros(sizeimg) ;
for i = 1:sizeimg(1,1)
    horlist = img(i,:) ;
    horlist_tmp = [0, horlist] ;
    horlist = [horlist,0] ;
    delta = horlist-horlist_tmp ;
    num = sum(abs(delta) == 1) ;
    if num >= horizon_threshold ;
        img_out(i,:) = img(i,:) ;
    end
end


