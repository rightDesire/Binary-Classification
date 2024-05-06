function out = mapFeature(X1, X2, num_features)
    out = ones(size(X1(:,1)));
    degree = floor(sqrt(2 * (num_features - 1)));
    if (degree > 1)
        for i = 1:degree
            for j = 0:i
                if (size(out, 2) < num_features) % Проверка на количество добавленных признаков
                    out(:, end+1) = (X1.^(i-j)).*(X2.^j);
                end
            end
        end
    else
        % Кастомный признак при degree = 0/1
        out(:, end+1) = X1;
        out(:, end+1) = X2;
        % out(:, end+1) = X1.^3+X2.^3;
        % out(:, end+1) = X1.^2.*X2.^2;
    end
end
