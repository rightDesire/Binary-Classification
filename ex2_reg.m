%% Инициализация
clear ; close all; clc

%% Инициализация данных
degree = 1;
lambda = 0;

% Настройки генерации примеров
num_examples_to_add = 1000;
max_noise_level = 0.15;

data = load('ex2data1.txt');

% % Генерация новых примеров на основе исходных
% [data] = generateExamples(init_data, ...
%     max_noise_level, num_examples_to_add);

%% Загрузка данных
[mAll, mTraining, mTest, X_train, y_train, X_test, y_test] = ... 
    loadData(data);

%% Расчет КФО при моделях с разными степенями полиномов
polynomialDegreeAnalysis(X_train, y_train, X_test, y_test, lambda);

%% Расчет КФО с разными значениями гиперпараметра регуляризации
regularizationAnalysis(X_train, y_train, X_test, y_test, degree);

%% Примеры (train, test)
% TRAIN
X_train = mapFeature(X_train(:,1), X_train(:,2), degree);

initial_theta = zeros(size(X_train, 2), 1);

options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta] = ...
        fminunc(@(t)(costFunctionReg(t, X_train, y_train, lambda)), ...
        initial_theta, options);

% Построение границы
plotDecisionBoundary(theta, X_train, y_train, degree);
hold on;
% Метки и легенда
xlabel('вибрация');
ylabel('непрерывность вращения');
% Указано в порядке построения
legend('Не исправен', 'Исправен', 'Граница решения');
% title(sprintf('Пример с %d степенью (train)', degree));
title(sprintf('График с обучающими примерами'));
hold off;

% TEST
X_test = mapFeature(X_test(:,1), X_test(:,2), degree);

initial_theta = zeros(size(X_test, 2), 1);

% Построение границы
plotDecisionBoundary(theta, X_test, y_test, degree);
hold on;
% Метки и легенда
xlabel('вибрация');
ylabel('непрерывность вращения');
% Указано в порядке построения
legend('Не исправен', 'Исправен', 'Граница решения');
% title(sprintf('Пример с %d степенью (test)', degree));
title(sprintf('График с тестовыми примерами'));
hold off;

%% Расчет качества обучающих примеров
% Вычисление точности на обучающем наборе
p_train = predict(theta, X_train);

fprintf('Точность обучения (train): %f\n', ...
    mean(double(p_train == y_train)) * 100);

% Вычисление точности на тестовом наборе
p_test = predict(theta, X_test);

fprintf('Точность обучения (test): %f\n', ...
    mean(double(p_test == y_test)) * 100);
