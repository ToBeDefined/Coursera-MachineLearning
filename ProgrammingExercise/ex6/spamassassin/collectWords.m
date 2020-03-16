function word_list = collectWords(email_contents)
word_list = {};

l = 0;

while ~isempty(email_contents)
    [str, email_contents] = strtok(email_contents, [' @$/#.-:&*+=[]?!(){},''">_<;%' char(10) char(13)]);
    str = regexprep(str, '[^a-zA-Z0-9]', '');

    try str = porterStemmer(strtrim(str)); 
    catch str = ''; continue;
    end;

    if length(str) < 1
        continue;
    end

    word_list{length(word_list)+1, 1} = str;

    if (l + length(str) + 1) > 78
        % fprintf('\n');
        l = 0;
    end
    % fprintf('%s ', str);
    l = l + length(str) + 1;

end

end
