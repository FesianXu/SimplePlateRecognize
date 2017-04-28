function saveRecognizeChars(plate_imgs, plate_types, root_path)
alp = 'alphabet\' ;
kanji = 'kanji\' ;
neg = 'neg\' ;
numbers = 'numbers\' ;
for i = 1:length(plate_imgs)
    if plate_types(i) >= 25 && plate_types(i) <= 34  %% numbers
        path = [root_path,numbers,getCharName(plate_types(i)),'\'] ;
        num_files = length(dir(path))-2 ;
        name = [getCharName(plate_types(i)),'-',num2str(num_files),'.png'] ;
        img_name = [path, name] ;
        imwrite(plate_imgs{i},img_name) ;
    elseif plate_types(i) >= 1 && plate_types(i) <= 24 % alphabet
        path = [root_path,alp,getCharName(plate_types(i)),'\'] ;
        num_files = length(dir(path))-2 ;
        name = [getCharName(plate_types(i)),'-',num2str(num_files),'.png'] ;
        img_name = [path, name] ;
        imwrite(plate_imgs{i},img_name) ;
    end % 
end



