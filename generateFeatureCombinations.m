function remaining_combinations = generateFeatureCombinations(num_features)
    if (num_features <= 3)
        % Если num_features <= 3, то возвращаем одну комбинацию: 1 1 1
        remaining_combinations = ones(1, 3);
    else
        % Иначе вычисляем общее количество комбинаций
        total_combinations = 2^(num_features - 3);
        % Создаем массив для хранения комбинаций
        remaining_combinations = zeros(total_combinations, num_features);
        
        % Генерация комбинаций
        for i = 1:total_combinations
            % Преобразуем числа от 0 до total_combinations - 1 в двоичный формат
            temp = dec2bin(i - 1, num_features - 3);
            % Добавляем первые три единицы
            binary_rep = ['1', '1', '1', temp];
            
            % Преобразование символов в числа и сохранение в массив
            remaining_combinations(i, :) = arrayfun(@(x) str2double(x), binary_rep);
        end
    end
end
