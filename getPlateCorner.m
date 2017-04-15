function [corner_list,list_new] = getPlateCorner(list, threh)
list_new = [list; list(1:threh,:)] ; % Ñ­»·ºó×º
list_len = length(list(:,1)) ;
delta = zeros(1,list_len) ;
for i = 1:list_len
    delta(i) = list_new(i+threh,1)-list_new(i,1) ;
end
delta_len = length(delta) ;
del2 = zeros(1,delta_len-1) ;
for i = 1:delta_len-1
    del2(i) = delta(i+1)-delta(i) ;
end
max_delta = max(delta) ;
delta_m = abs(delta) >= max_delta-1 ;
del2_m = del2 == 0 ;
del2_m(length(delta_m)) = 0;
delta_m = [0,delta_m] ;
del2_m = [0,del2_m] ;
tina = delta_m.*del2_m ;
tina_len = length(tina) ;
corner_list = zeros(tina_len-1) ;
for i = 1:tina_len-1
    corner_list(i) = tina(i+1)-tina(i) ;
end

