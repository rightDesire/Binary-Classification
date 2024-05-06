function plotDecisionBoundaryAndAccuracy(X_train, y_train, X_test, y_test, num_features, lambda)
    X_train = mapFeature(X_train(:,1), X_train(:,2), num_features);
    X_test = mapFeature(X_test(:,1), X_test(:,2), num_features);
    
    initial_theta = zeros(size(X_train, 2), 1);
    
    options = optimset('GradObj', 'on', 'MaxIter', 400);
    [theta] = ...
            fminunc(@(t)(costFunctionReg(t, X_train, y_train, lambda)), ...
            initial_theta, options);
    
    regularizationAnalysis(X_train, y_train, ...
        X_test, y_test);
    
    %% Расчет качества обучающих примеров
    fprintf('Модель с %d признаками, при лямбде %g:\n', num_features, lambda);

    % Вычисление точности на обучающем наборе
    p_train = predict(theta, X_train);

    fprintf('Точность обучения (train): %f\n', ...
        mean(double(p_train == y_train)) * 100);
    
    % Вычисление точности на тестовом наборе
    p_test = predict(theta, X_test);
    
    fprintf('Точность обучения (test): %f\n', ...
        mean(double(p_test == y_test)) * 100);
    
    %% Построение графиков
    plotDecisionBoundary(theta, X_train, y_train, size(X_train, 2));
    hold on;
    title(sprintf('Модель с %d признаками, \n при лямбде %g, \n с обучающими примерами', num_features, lambda));
    xlabel('Тест 1');
    ylabel('Тест 2');
    legend('y = 1', 'y = 0', 'Граница решения');
    hold off;
    
    plotDecisionBoundary(theta, X_test, y_test, size(X_test, 2));
    hold on;
    title(sprintf('Модель с %d признаками, \n при лямбде %g, \n с тестовыми примерами', num_features, lambda));
    xlabel('Тест 1');
    ylabel('Тест 2');
    legend('y = 1', 'y = 0', 'Граница решения');
    hold off;
end
