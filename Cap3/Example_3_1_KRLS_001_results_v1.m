close all;
clear all;
clc;

% Parâmetros do algoritmo KRLS
lambda = 0.40; % Fator de esquecimento
I_max = 65; % Tamanho máximo do dicionário
K = 10000;
I = 1;

kernel_type = "gaussian";
kernel_param = [0.1 1 0.1]; % Parâmetro do kernel gaussiano

% Sinal de entrada (exemplo)
signal = randn(1, K); % Sinal aleatório

% Inicialização
d = zeros(1, I_max); % Dicionário
alpha = zeros(1, I_max); % Coeficientes do modelo
mse = zeros(1, length(signal)); % Para armazenar o MSE em cada iteração
mse_k = zeros(1, length(signal)); % Para armazenar o MSE em cada iteração
e_k = zeros(1, length(signal)); % Para armazenar o erro em cada iteração
kernel_values = 0;

% Algoritmo KRLS
for k = 1:length(signal)
    if k > 1
        % exp(-sum(((d(1:min(k-1, I_max)) - signal(k)).^2), 1) / (2 * 0.1^2))
        kernel_values = calculate_kernel_values(d(1:min(k-1, I_max)), signal(k), kernel_type, kernel_param);
        alpha_k = alpha(1:min(k-1, I_max)) .* kernel_values;
        alpha_k_sum = sum(alpha_k);
        alpha(k) = lambda * alpha_k_sum + (1 - lambda) * signal(k) * kernel_values / (kernel_values * alpha_k_sum + lambda);
    else
        alpha(k) = signal(k); % Primeira iteração: alpha inicializado com o primeiro sinal
    end
    
    % Atualiza o dicionário
    [max_kernel_value, l_max] = max(abs(kernel_values));
    
    if k <= I_max
        I = I + 1;
        d(k) = signal(k);
    else
        [~, l_min] = min(abs(d - signal(k))); % Encontra o índice do elemento mais distante
        d(l_min) = signal(k); % Substitui o elemento mais distante pelo novo sinal
    end
    
    % Calcula o erro
    y_pred = calculate_kernel_values(d(1:min(k, I_max)), signal(k), kernel_type, kernel_param) * alpha(1:min(k, I_max))';
    e = signal(1:k) - y_pred;
    
    e_k(k) = signal(k) - y_pred;

    % Calcula o MSE
    mse_k(k) = mean(e_k(k).^2);
    
    % Calcula o MSE
    mse(k) = mean(e.^2);
end

y_pred_result = zeros(1, I_max); % Previsão futura
for k = 1:length(signal)
    y_pred_result(k) = calculate_kernel_values(d(1:min(k, I_max)), signal(k), kernel_type, kernel_param) * alpha(1:min(k, I_max))';
end

% Plot do gráfico MSE em K iterações em dB
figure;
plot(1:length(signal), 10*log10(mse));
xlabel('Iterações (k)');
ylabel('MSE (dB)');
title('Evolução do MSE ao longo das iterações (dB)');

figure;
plot(1:length(signal), 10*log10(mse_k));
xlabel('Iterações (k)');
ylabel('MSE (dB)');
title('Evolução do MSE_k ao longo das iterações (dB)');

% Plot do gráfico MSE em K iterações em dB
figure;
plot(1:length(signal), 10*log10(abs(e_k)));
xlabel('Iterações (k)');
ylabel('e(k) (dB)');
title('Evolução do e(k) ao longo das iterações (dB)');

figure;
plot(signal,"r")
hold on
plot(y_pred_result,"g")
% plot(y,"b")
hold off
title("Sinal","Interpreter","latex");
% legend("d(k)","g(k)","g*x");
legend("signal(k)","d(k)");
% legend("d(k)");
% xlim([0 100]);

% Função para calcular os valores do kernel
function kernel_values = calculate_kernel_values(x, y, kernel_type, kernel_param)
    switch kernel_type
        case 'cosine'
            kernel_values = diag(x' * y) ./ (sqrt(sum(x.^2, 1))' * sqrt(sum(y.^2, 1)))';
        case 'sigmoid'
            kernel_values = tanh(kernel_param(1) * (x' * y) + 1);
        case 'polynomial'
            kernel_values = (kernel_param(1)*x' * y + kernel_param(2)).^kernel_param(3);
        case 'gaussian'
            kernel_values = exp(-sum((x - y).^2, 1) / (2 * kernel_param(1)^2));
        case 'laplacian'
            kernel_values = exp(-sum(abs(x - y), 1) / kernel_param(1));
        otherwise
            error('Tipo de kernel não suportado.');
    end
end
