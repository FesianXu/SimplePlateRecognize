function charout = getCharName(num)
%% ������û��I����ַ���û��O����ַ�
if num >=1 && num <= 8
    begin = abs('A') ;
    charout = char(begin+num-1) ;
elseif num >= 9 && num <= 13
    begin = abs('J') ;
    charout = char(begin+num-9) ;
elseif num >= 14 && num <= 24 %% alphabet part
    begin = abs('P') ;
    charout = char(begin+num-14) ;
elseif num >= 25 && num <= 34 %% number part
    begin = abs('0');
    charout = char(begin+num-25) ;
end
