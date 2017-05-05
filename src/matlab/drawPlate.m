function [rimg, num] = drawPlate(img, con_cell)
con_size = size(con_cell) ;
con_size = con_size(1,2) ;
rimg = img ;
for i = 1:con_size
    y_max = max(con_cell{i}(:,1));
    y_min = min(con_cell{i}(:,1));
    x_max = max(con_cell{i}(:,2)) ;
    x_min = min(con_cell{i}(:,2)) ;
    rimg = insertShape(rimg, 'Line',[x_min,y_min,x_max,y_min], 'LineWidth', 5, 'Color', 'red') ;
    rimg = insertShape(rimg, 'Line',[x_min,y_min,x_min,y_max], 'LineWidth', 5, 'Color', 'red') ;
    rimg = insertShape(rimg, 'Line',[x_min,y_max,x_max,y_max], 'LineWidth', 5, 'Color', 'red') ;
    rimg = insertShape(rimg, 'Line',[x_max,y_min,x_max,y_max], 'LineWidth', 5, 'Color', 'red') ;
end
num = con_size ;