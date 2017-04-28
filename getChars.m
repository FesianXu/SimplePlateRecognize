function [charset] = getChars(plate_img)
%% ���õͶ˵ķ���������������Ϣ�ж��ַ�λ�ã�������������󣬿��Բ��ûع����ϻ������ڵķ�ʽ��
%%% charset = {char_imgs, center_set, char_numbers}
chars_least_length = 20 ;
chars_down = 1.5 ;
chars_up = 5 ;
debug = 'on' ;
%% ɾ�����������ַ���
imgn_con = bwboundaries(plate_img,8, 'noholes') ;
char_con = {} ;
inner = 1 ;
for i = 1:length(imgn_con)
    if length(imgn_con{i}(:,1)) < chars_least_length
        continue ;
    end
    drow = max(imgn_con{i}(:,1))-min(imgn_con{i}(:,1)) ;
    dcol = max(imgn_con{i}(:,2))-min(imgn_con{i}(:,2)) ;
    pval = drow/dcol ;
    if pval > chars_down && pval < chars_up && drow > 10 && dcol > 5 && drow < 40 && dcol < 20
        char_con{inner} = imgn_con{i} ; 
        inner = inner+1 ;
    end
end
%% plot 
if debug == 'on'
    figure(2)
    subplot(2,1,1)
    imshow(plate_img)
    hold on
    for i = 1:length(char_con)
        plot(char_con{i}(:,2),char_con{i}(:,1),'r*')
    end
end

%% get center location
plate_width = length(plate_img(1,:)) ;
origin_plate_center = zeros(length(char_con),2) ;
boxing_set = zeros(length(char_con),2) ;
boxout_set = zeros(length(char_con),4) ;
for i = 1:length(char_con)
    center_row = (max(char_con{i}(:,1))+min(char_con{i}(:,1)))/2 ;
    center_col = (max(char_con{i}(:,2))+min(char_con{i}(:,2)))/2 ;
    boxing_set(i,:) = [max(char_con{i}(:,1))-min(char_con{i}(:,1)), max(char_con{i}(:,2))-min(char_con{i}(:,2))] ;
    origin_plate_center(i,:) = [uint16(center_row), uint16(center_col)] ;
    boxout_set(i,:) = [max(char_con{i}(:,1)), min(char_con{i}(:,1)), max(char_con{i}(:,2)), min(char_con{i}(:,2))] ;
    if debug == 'on'
        hold on
        plot(center_col,center_row,'b*')
    end
end
type_list = decideCharType(origin_plate_center(:,2)/plate_width) ;
%% ������ô�����������һ���ַ��Ķ�������ظ���ʱ����Ҫ�غ��ⲿ�֣���ֹ��������
%%% �ϲ�chars_set��type_list
chars_set_combined = {} ;
inner_loop = 1 ;
for i = 1:7
    loc = find(type_list == i) ;
    tmpchars = [] ;
    for j = 1:length(loc)
        tmpchars = [tmpchars; char_con{loc(j)}];
    end
    if isempty(tmpchars)
        continue ;
    end
    chars_set_combined{inner_loop} = tmpchars ;
    inner_loop = inner_loop+1 ;
end
origin_plate_center = zeros(length(chars_set_combined),2) ;
for i = 1:7
    loc = find(type_list == i) ;
    if length(loc) > 1
        for j = length(loc):-1:2
            type_list(loc(j)) = [] ;
        end
    end
end
for i = 1:length(chars_set_combined)
    center_row = (max(chars_set_combined{i}(:,1))+min(chars_set_combined{i}(:,1)))/2 ;
    center_col = (max(chars_set_combined{i}(:,2))+min(chars_set_combined{i}(:,2)))/2 ;
    origin_plate_center(i,:) = [uint16(center_row), uint16(center_col)] ;
end
%% poly approximate
if type_list(1) == 1 && length(type_list) > 2
    x_fit = origin_plate_center(2:end,2) ;
    y_fit = origin_plate_center(2:end,1) ;
else
    x_fit = origin_plate_center(:,2) ;
    y_fit = origin_plate_center(:,1) ;
end
[P,~]=polyfit(x_fit,y_fit,1) ;
x_cor = 1:plate_width ;
y_cor = uint16(P(1).*x_cor+P(2)) ;
if debug == 'on'
    hold on
    plot(x_cor,y_cor,'g*')
end

%% reget
boxmsg_set = [] ; % [center_row,center_col,row_box,col_box,max_row,min_row,max_col,min_col,char_number]
all_type = 1:7 ;
maxbox_row = max(boxing_set(:,1)) ;
maxbox_col = max(boxing_set(:,2)) ;
diff_type = setdiff(all_type, type_list) ;
for i = 1:length(type_list)
    boxmsg_set(type_list(i),1:2) = origin_plate_center(i,:) ;
    boxmsg_set(type_list(i),3:4) = boxing_set(i,:) ;
    boxmsg_set(type_list(i),5:8) = boxout_set(i,:) ;
    boxmsg_set(type_list(i),9) = type_list(i) ;
end
sum_deltacol = 0 ;
sum_deltatype = 0 ;
for i = 1:length(type_list)-1
    if type_list(i) == 3 || type_list(i) == 4 || type_list(i) == 5 || type_list(i) == 6 || type_list(i) == 7
        sum_deltacol = origin_plate_center(i+1,2)-origin_plate_center(i,2)+sum_deltacol ;
        sum_deltatype = type_list(i+1)-type_list(i)+sum_deltatype ;
    end
end
dist = [uint16(sum_deltacol*1.386/sum_deltatype),uint16(sum_deltacol/sum_deltatype)] ;
%%% �û��get����chars
for i = 1:length(diff_type)
    actval = type_list-diff_type(i) ;
    absval = abs(actval) ;
    [~, ind] = min(absval) ;
    ind_type = type_list(ind) ;
    
    if ind_type > 2 % ��ê���ұ�
        if diff_type(i) > ind_type
            delta = diff_type(i)-ind_type ;
            boxmsg_set(diff_type(i),2) = boxmsg_set(ind_type,2)+delta*dist(2) ;
        else
            delta = ind_type-diff_type(i) ;
            if diff_type(i) <= 2
                boxmsg_set(diff_type(i),2) = boxmsg_set(ind_type,2)-dist(1)-(delta-1)*dist(2);
            else
                boxmsg_set(diff_type(i),2) = boxmsg_set(ind_type,2)-delta*dist(2);
            end
        end
    else % ��ê�����
        if diff_type(i) > ind_type % ��Ҫ��ȫ���ַ����������Ҳ�
            delta = diff_type(i)-ind_type ;
            if diff_type(i) > 2
                boxmsg_set(diff_type(i),2) = boxmsg_set(ind_type,2)+dist(1)+(delta-1)*dist(2);
            elseif diff_type(i) == 2
                boxmsg_set(diff_type(i),2) = boxmsg_set(ind_type,2)+dist(2);
            end
        else % ���
            boxmsg_set(diff_type(i),2) = boxmsg_set(ind_type,2)-dist(2)+abs(P(1))*50 ;
        end
    end
    boxmsg_set(diff_type(i),1) = y_cor(boxmsg_set(diff_type(i),2)) ;
    boxmsg_set(diff_type(i),9) = diff_type(i) ;
end
for i = 1:length(boxmsg_set(:,1))
     boxmsg_set(i,1) = y_cor(boxmsg_set(i,2)) ;
end

if debug == 'on'
    for i = 1:length(boxmsg_set(:,1))
        hold on
        plot(boxmsg_set(i,2),boxmsg_set(i,1),'r*')
    end
end

%% plot 
if debug == 'on'
    for i = 1:length(type_list)
        hold on
        rectangle('Position',[boxmsg_set(type_list(i),8) boxmsg_set(type_list(i),6) boxmsg_set(type_list(i),4) boxmsg_set(type_list(i),3)],'EdgeColor','r')
    end
    hold on
    for i = 1:length(chars_set_combined)
        plot(chars_set_combined{i}(:,2),chars_set_combined{i}(:,1),'y*')
    end
end

charset = {} ;
max_boxrow = max(boxmsg_set(:,3)) ;
max_boxcol = max(boxmsg_set(:,4)) ;
for i = 1:length(boxmsg_set(:,1))
    row_max = boxmsg_set(i,1)+uint16(max_boxrow/2)+1 ;
    row_min = boxmsg_set(i,1)-uint16(max_boxrow/2)-1 ;
    col_max = boxmsg_set(i,2)+uint16(max_boxcol/2)+1 ;
    col_min = boxmsg_set(i,2)-uint16(max_boxcol/2)-1 ;
    if row_min < 1
        row_min = 1 ;
    end
    if col_min < 1
        col_min = 1 ;
    end
    if col_max > length(plate_img(1,:))
        col_max = length(plate_img(1,:)) ;
    end
    if row_max > length(plate_img(:,1))
        row_max = length(plate_img(:,1)) ;
    end
    charset{i} = plate_img(row_min:row_max, col_min:col_max) ;
end

