function featureCombinationsAnalysis(X_train, y_train, X_test, y_test, ...
    num_features, lambda)
    X_train = mapFeature(X_train(:,1), X_train(:,2), num_features);
    X_test = mapFeature(X_test(:,1), X_test(:,2), num_features);

    combinations = generateFeatureCombinations(num_features);
    
    for i = 1:size(combinations, 1)
        fprintf('Модель %d:\n', i);
        
        % Получаем индексы признаков для текущей модели
        feature_i = combinations(i, :);
        
        % Выводим состав модели
        disp(feature_i);

        % Создаем новые данные на основе выбранных признаков
        X_train_subset = X_train .* feature_i;
        X_test_subset = X_test .* feature_i;

        initial_theta = zeros(size(X_train_subset, 2), 1);
    
        options = optimset('GradObj', 'on', 'MaxIter', 400);
        [theta] = ...
                fminunc(@(t)(costFunctionReg(t, X_train_subset, y_train, lambda)), ...
                initial_theta, options);
        
        regularizationAnalysis(X_train_subset, y_train, ...
            X_test_subset, y_test);

        %% Расчет качества обучающих примеров
        % Вычисление точности на обучающем наборе
        p_train = predict(theta, X_train_subset);
        
        fprintf('Точность обучения (train): %f\n', ...
            mean(double(p_train == y_train)) * 100);
        
        % Вычисление точности на тестовом наборе
        p_test = predict(theta, X_test_subset);
        
        fprintf('Точность обучения (test): %f\n', ...
            mean(double(p_test == y_test)) * 100);
        
        %% Построение графиков
        plotDecisionBoundary(theta, X_train_subset, y_train, size(X_train_subset, 2));
        hold on;
        title(sprintf('Модель %d с обучающими примерами', i));
        xlabel('вибрация');
        ylabel('непрерывность вращения');
        legend('Не исправен', 'Исправен', 'Граница решения');
        hold off;

        plotDecisionBoundary(theta, X_test_subset, y_test, size(X_test_subset, 2));
        hold on;
        title(sprintf('Модель %d с тестовыми примерами', i));
        xlabel('вибрация');
        ylabel('непрерывность вращения');
        legend('Не исправен', 'Исправен', 'Граница решения');
        hold off;
    end
end