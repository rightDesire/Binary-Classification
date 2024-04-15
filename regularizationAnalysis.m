function regularizationAnalysis(X_train, y_train, X_test, y_test, degree)
    options = optimset('GradObj', 'on', 'MaxIter', 400);
    lambda_values = 0:100;

    % Создание массива для хранения ошибок
    J_train = zeros(size(lambda_values));
    J_test = zeros(size(lambda_values));

    % Цикл по различным значениям lambda
    for i = 1:length(lambda_values)
        lambda = lambda_values(i);

        % Создание признаков для фиксированной степени полинома
        X_train_poly = mapFeature(X_train(:,1), X_train(:,2), degree);
        X_test_poly = mapFeature(X_test(:,1), X_test(:,2), degree);

        % Инициализация параметров подгонки
        initial_theta = zeros(size(X_train_poly, 2), 1);

        % Оптимизация
        [theta] = ...
            fminunc(@(t)(costFunctionReg(t, X_train_poly, y_train, lambda)), ...
            initial_theta, options);

        % Вычисление стоимости для обучающей выборки
        J_train(i) = costFunctionReg(theta, X_train_poly, y_train, lambda);

        % Вычисление стоимости для тестовой выборки
        J_test(i) = costFunctionReg(theta, X_test_poly, y_test, lambda);

        fprintf("СКО, при lambda %f, с обучающими" + ...
            " и тренировочными примерами соответственно: %f ; %f\n", lambda, ...
            J_train(i), J_test(i));
    end

    % Построение графика
    figure;
    hold on;
    plot(lambda_values, J_train, '-b', 'LineWidth', 2, 'MarkerSize', 8);
    plot(lambda_values, J_test, '-r', 'LineWidth', 2, 'MarkerSize', 8);
    xlabel('lambda');
    ylabel('Ошибка');
    legend('J train', 'J test');
    title('График ошибок для обучающего и тестового наборов');
    hold off;
end
