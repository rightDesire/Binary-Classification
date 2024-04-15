function p = predict(theta, X)
    m = size(X, 1);
    p = zeros(m, 1);
    probabilities = sigmoid(X * theta);
    p = probabilities >= 0.5;
end
