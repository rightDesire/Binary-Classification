function [mAll, mTraining, mTest, X_train, y_train, X_test, y_test] = ... 
    loadData(data)

    % Размеры данных
    mAll = size(data, 1);
    mTraining = round(mAll * 0.7); % 70% на обучение
    mTest = mAll - mTraining;
    
    % Генерация случайных индексов для перемешивания данных
    random_indices = randperm(mAll);
    
    % Перемешивание данных
    shuffled_data = data(random_indices, :);
    
    % Разделение перемешанных данных на обучающую и тестовую выборки
    X_train = shuffled_data(1:mTraining, 1:2);
    y_train = shuffled_data(1:mTraining, 3);
    
    X_test = shuffled_data(mTraining+1:end, 1:2);
    y_test = shuffled_data(mTraining+1:end, 3);
end
