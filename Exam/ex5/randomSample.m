function [X_randoms, y_randoms] = randomSample(X, y, random_len)

index = randperm(size(X, 1), random_len);
X_randoms = X(index, :);
y_randoms = y(index, :);

end
