
% Parâmetros do algoritmo KLMS
mu = 0.75; % Taxa de aprendizado
gamma_c = 0.9; % Limiar para adicionar um novo ponto ao dicionário
I_max = 5; % Tamanho máximo do dicionário

N = 10;

K = 100;

% Sinal de entrada (exemplo)
d = randn(1, K); % Sinal aleatório

% Função kernel escolhida
kernel_type = 'polynomial'; % Escolha entre 'cosine', 'sigmoid', 'polynomial', 'gaussian', 'laplacian', ou outras
kernel_param = 0.01; % Parâmetro do kernel (pode variar dependendo do tipo de kernel)

kernel_lib_temp = zeros(I_max,1);

l_max = 1;
I = l_max;

% Inicialização
x = zeros(length(d), I_max); % Dicionário
e = zeros(1, length(d)); % Erro
l_0 = ones(1, length(d)); % Índices iniciais dos pontos no dicionário
mse = zeros(1, length(d)); % Para armazenar o MSE em cada iteração
g = zeros(1, length(d)); % 

% Algoritmo KLMS
for k = 1:length(d)
    if k > 1

%         for l = 1:lmax
%             kernel_lib_temp(l) = kernel_using(x(k-lmax:-1:k-lmax-N,1),x(k:-1:k-N),C,D,E);
%             g = 2*mu_g*e_l(l)*kernel_lib_temp(l) + 2*mu_g*kernel_using(x(k:-1:k-N),x(k:-1:k-N),C,D,E);
%         end
%         g = sum(g_sum);
%         e(k) = d(k)-g(k);

%         size(x(k-l_max+1:-1:1,1))
%         size(x(k:-1:1))
        
        if k <= N+1
            for l = 1:l_max
                kernel_values = calculate_kernel_values(x(k-l_max+1:-1:1,1), x(k:-1:1,1), kernel_type, kernel_param);
                kernel_lib_temp(l) = kernel_values;
                g(k-1) = 2*mu*e(l)*kernel_lib_temp(l) + 2*mu*calculate_kernel_values(x(k:-1:1,1),x(k:-1:1,1), kernel_type, kernel_param);
            end
        else
            for l = 1:l_max
                kernel_values = calculate_kernel_values(x(k-l_max:-1:k-l_max-N,I),x(k:-1:k-N,I), kernel_type, kernel_param);
                kernel_lib_temp(l) = kernel_values;
                g(k-1) = 2*mu*e(l)*kernel_lib_temp(l) + 2*mu*calculate_kernel_values(x(k:-1:k-N,I),x(k:-1:k-N,I), kernel_type, kernel_param);
            end
        end

%         kernel_values = calculate_kernel_values(x(:,1:k-1), x(:,k), kernel_type, kernel_param);
%         g(k) = 2 * mu * sum(repmat(e(1:k-1), 1, size(kernel_values, 1)) .* kernel_values, 1) + 2 * mu * sum(kernel_values);
        e(k) = d(k) - g(k-1);
        
        [max_kernel_value, l_max] = max(abs(kernel_values));
        
        if max_kernel_value <= gamma_c
            I = sum(l_0(1:k-1) > 0) + 1;
            if I <= I_max
                l_0(k+1) = k + 1;
                x(:,I) = d(k) * ones(length(d), 1); % Adiciona x(k) ao dicionário
            else
                [~, l_max] = max(abs(x(:, l_0 == k)));
                x(:, l_max) = d(k) * ones(length(d), 1); % Substitui x(l_max) por x(k) no dicionário
            end
        else
            e(l_max) = e(l_max) + mu * e(k);
        end
    end
    mse(k) = mean(e(1:k).^2);
end

% Plot do gráfico MSE em K iterações em dB
figure;
plot(1:length(d), 10*log10(mse));
xlabel('Iterações (k)');
ylabel('MSE (dB)');
title('Evolução do MSE ao longo das iterações (dB)');

% Plot do gráfico MSE em K iterações em dB
figure;
plot(1:length(d), 10*log10(abs(e)));
xlabel('Iterações (k)');
ylabel('e(k) (dB)');
title('Evolução do e(k) ao longo das iterações (dB)');

figure;
plot(d,"g")
hold on
plot(g,"r")
% plot(y,"b")
hold off
title("Sinal","Interpreter","latex");
% legend("d(k)","g(k)","g*x");
legend("d(k)","g(k)");
% legend("d(k)");
% xlim([0 100]);

% Função para calcular os valores do kernel
function kernel_values = calculate_kernel_values(x, y, kernel_type, kernel_param)
    switch kernel_type
        case 'cosine'
            kernel_values = diag(x' * y) ./ (sqrt(sum(x.^2, 1))' * sqrt(sum(y.^2, 1)))';
        case 'sigmoid'
            kernel_values = tanh(kernel_param * (x' * y) + 1);
        case 'polynomial'
            kernel_values = (x' * y + 1).^kernel_param;
        case 'gaussian'
            kernel_values = exp(-sum((x - y).^2, 1) / (2 * kernel_param^2));
        case 'laplacian'
            kernel_values = exp(-sum(abs(x - y), 1) / kernel_param);
        otherwise
            error('Tipo de kernel não suportado.');
    end
end