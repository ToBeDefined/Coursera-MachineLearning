function [vocabList, vocabTimes] = createVocabList(EmailContents)

vocabList = {};
vocabTimes = [];
fprintf('\nEmail Counts: %d\n', size(EmailContents, 1));
for i = 1:size(EmailContents, 1)
    fprintf('>> collectWords in Email: %d\n', i);
    word_list = collectWords(EmailContents(i,:));
    for j = 1:length(word_list)
        word = word_list{j};
        wordIdx = find(strcmp(word, vocabList), 1); 
        if isempty(wordIdx)
            vocabList{length(vocabList)+1, 1} = word;
            vocabTimes = [vocabTimes; 1];
        else
            vocabTimes(wordIdx) += 1;
        endif
    endfor
endfor

[vocabTimes, indices] = sort(vocabTimes, "descend");
vocabList = vocabList(indices);

vocabList{1:20}
vocabTimes(1:20)

end
