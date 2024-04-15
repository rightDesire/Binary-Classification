function polynomialDegreeAnalysis(X_train, y_train, X_test, y_test, lambda)
    options = optimset('GradObj', 'on', 'MaxIter', 400);

    % Создание массива для хранения ошибок
    d_values = 1:5; % степени полинома
    J_train = zeros(size(d_values));
    J_test = zeros(size(d_values));

    % Цикл по различным степеням полинома
    for i = 1:length(d_values)
        % Создание признаков для данной степени полинома
        X_train_poly = mapFeature(X_train(:,1), X_train(:,2), d_values(i));
        X_test_poly = mapFeature(X_test(:,1), X_test(:,2), d_values(i));

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

        fprintf("СКО, при степени полинома %f, с обучающими" + ...
            " и тренировочными примерами соответственно: %f ; %f\n", ...
            i, J_train(i), J_test(i));
    end

    % Построение графика
    figure;
    hold on;
    plot(d_values, J_train, '-b', 'LineWidth', 2, 'MarkerSize', 8);
    plot(d_values, J_test, '-r', 'LineWidth', 2, 'MarkerSize', 8);
    xlabel('Степень полинома d');
    ylabel('Ошибка');
    legend('J train', 'J test');
    title('График ошибок для обучающего и тестового наборов');
    hold off;
end
