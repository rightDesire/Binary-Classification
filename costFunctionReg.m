function [J, grad] = costFunctionReg(theta, X, y, lambda)
    m = length(y); % количество примеров в обучающем наборе

    % Вычисление гипотезы
    hypothesis = sigmoid(X * theta);

    % Вычисление стоимости
    J = (1 / m) * sum(-y .* log(hypothesis) - (1 - y) ...
        .* log(1 - hypothesis)) + (lambda / (2 * m)) ...
        * sum(theta(2:end).^2);

    % Вычисление градиента
    grad = (1 / m) * X' * (hypothesis - y);
    % Регуляризация всех элементов градиента, кроме theta(1)
    grad(2:end) = grad(2:end) + (lambda / m) * theta(2:end);

end
