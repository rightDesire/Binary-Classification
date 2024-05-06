function polynomialDegreeAnalysis(X_train, y_train, X_test, y_test, lambda)
    options = optimset('GradObj', 'on', 'MaxIter', 400);

    % Создание массива для хранения ошибок
    degrees = 1:20; % кол-во степеней полиномов
    J_train = zeros(size(degrees));
    J_test = zeros(size(degrees));

    % Цикл по различным степеням полиномов
    for degree = degrees
        % Создание признаков для данной степени полинома
        X_train_poly = mapFeatureDegree(X_train(:,1), X_train(:,2), degree);
        X_test_poly = mapFeatureDegree(X_test(:,1), X_test(:,2), degree);

        % Инициализация параметров подгонки
        initial_theta = zeros(size(X_train_poly, 2), 1);

        % Оптимизация
        [theta] = ...
            fminunc(@(t)(costFunctionReg(t, X_train_poly, y_train, lambda)), ...
            initial_theta, options);

        % Вычисление стоимости для обучающей выборки
        J_train(degree) = costFunctionReg(theta, X_train_poly, y_train, lambda);

        % Вычисление стоимости для тестовой выборки
        J_test(degree) = costFunctionReg(theta, X_test_poly, y_test, lambda);

        fprintf("СКО, при степени полинома %d, с обучающими" + ...
            " и тренировочными примерами соответственно: %f ; %f\n", ...
            degree, J_train(degree), J_test(degree));
    end

    % Построение графика
    figure;
    hold on;
    plot(degrees, J_train, '-b', 'LineWidth', 2, 'MarkerSize', 8);
    plot(degrees, J_test, '-r', 'LineWidth', 2, 'MarkerSize', 8);
    xlabel('Степень полинома');
    ylabel('Ошибка');
    legend('J train', 'J test');
    title({'Зависимость ошибки от степени полинома:', 'обучающий и тестовый наборы'});
    hold off;
end
