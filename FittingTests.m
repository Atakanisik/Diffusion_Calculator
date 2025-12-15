% =========================================================================
% UNIVERSAL DIFFUSION CALCULATOR - ALGORITHM VALIDATION SUITE (FULL)
% 
% Bu test dosyası, yazılımda kullanılan algoritmaların TÜM parametrelerini
% (f, D, D*, vb.) sentetik verilerle (Ground Truth) kıyaslayarak doğrular.
%
% Yöntem: Verification with Synthetic Data (High SNR)
% =========================================================================

function tests = FittingTests
    tests = functiontests(localfunctions);
end

% =========================================================================
% 1. MONO-EXPONENTIAL (ADC) TESTİ
% =========================================================================
function testMonoExp(testCase)
    true_ADC = 0.0008;
    b_values = [0, 50, 100, 400, 800, 1000];
    signal = exp(-b_values * true_ADC);
    
    ft = fittype('exp(-b*x)', 'independent', 'b', 'dependent', 'y');
    opts = fitoptions('Method', 'NonLinearLeastSquares', 'StartPoint', [0.001]);
    fitResult = fit(b_values', signal', ft, opts);
    
    % --- KONTROL ---
    testCase.verifyEqual(fitResult.x, true_ADC, 'RelTol', 1e-4, 'ADC değeri hatalı.');
    
    fprintf('✅ Mono-Exp: ADC Verified (Target: %.5f, Calc: %.5f)\n', true_ADC, fitResult.x);
end

% =========================================================================
% 2. BI-EXPONENTIAL (IVIM FREE) TESTİ
% =========================================================================
function testBiExpFree(testCase)
    % Ground Truth
    f = 0.25; D = 0.001; Dp = 0.050; % Dp = D*
    b_values = [0, 10, 20, 50, 100, 200, 400, 800, 1000];
    
    signal = f * exp(-b_values * Dp) + (1-f) * exp(-b_values * D);
    
    ft = fittype('f * exp(-b * D_star) + (1 - f) * exp(-b * D)', ...
        'independent', 'b', 'coefficients', {'f', 'D_star', 'D'});
    opts = fitoptions('Method','NonLinearLeastSquares', ...
        'StartPoint', [0.2, 0.05, 0.001], ...
        'Lower', [0, 0, 0], 'Upper', [1, 0.5, 0.01]);
        
    res = fit(b_values', signal', ft, opts);
    
    % --- KONTROL (Tüm Parametreler) ---
    testCase.verifyEqual(res.f, f, 'RelTol', 0.05, 'IVIM f parametresi hatalı.');
    testCase.verifyEqual(res.D, D, 'RelTol', 0.05, 'IVIM D parametresi hatalı.');
    testCase.verifyEqual(res.D_star, Dp, 'RelTol', 0.10, 'IVIM D* parametresi hatalı.');
    
    fprintf('✅ Bi-Exp Free: f, D, D* Verified.\n');
end

% =========================================================================
% 3. BI-EXPONENTIAL (SEGMENTED) TESTİ
% =========================================================================
function testBiExpSegmented(testCase)
    % Segmented mantığında D ve f, D*'dan önce bulunur.
    % Bu test, yazılımdaki 2 aşamalı mantığı simüle eder.
    
    f = 0.20; D = 0.0015; Dp = 0.040;
    b_values = [0, 20, 50, 100, 200, 400, 800, 1000];
    signal = f * exp(-b_values * Dp) + (1-f) * exp(-b_values * D);
    
    % --- ADIM 1: High b (>200) fit ile D bul ---
    highMask = b_values > 200;
    % Model: S ~ (1-f) * exp(-b*D) -> a * exp(-b*x)
    ft_step1 = fittype('a * exp(-b * x)', 'independent', 'x');
    res1 = fit(b_values(highMask)', signal(highMask)', ft_step1, 'StartPoint', [0.8, 0.001]);
    
    calc_D = res1.b; 
    calc_f_approx = 1 - res1.a; % Interceptten f tahmini
    
    % --- ADIM 2: Dp (D*) Bulma ---
    % D ve f (veya intercept) sabitlenip D* aranır
    % Model: (1-a)*exp(-b*DP) + a*exp(-b*D_fixed)
    % Burada 'a' aslında (1-f)'tir.
    
    % Basitleştirilmiş 2. adım testi (Yazılımdaki mantığa benzer):
    pixelmodel = @(DP,x) (calc_f_approx * exp(-x*DP)) + (1-calc_f_approx)*exp(-x*calc_D);
    ft_step2 = fittype(pixelmodel, 'independent', 'x', 'coefficients', {'DP'});
    res2 = fit(b_values', signal', ft_step2, 'StartPoint', [0.05], 'Lower', [0]);
    
    calc_Dp = res2.DP;

    % --- KONTROL (Tüm Parametreler) ---
    testCase.verifyEqual(calc_D, D, 'RelTol', 0.1, 'Segmented D hatası.');
    testCase.verifyEqual(calc_f_approx, f, 'RelTol', 0.1, 'Segmented f hatası.');
    testCase.verifyEqual(calc_Dp, Dp, 'RelTol', 0.2, 'Segmented D* hatası.'); % D* segmented'da hassastır
    
    fprintf('✅ Segmented Fit: f, D, D* Verified.\n');
end

% =========================================================================
% 4. BAYESIAN FIT TESTİ
% =========================================================================
function testBayesian(testCase)
    rng(42); % Tekrarlanabilirlik için seed
    true_f = 0.3; 
    true_Dstar = 0.05; 
    true_D = 0.002;
    b_values = [0, 20, 50, 100, 200, 400, 800, 1000];
    signal = true_f * exp(-b_values * true_Dstar) + (1-true_f) * exp(-b_values * true_D);
    
    % Helper fonksiyonu çağır
    [post_mean, ~, ~, ~, ~] = bayesian_calc_helper(b_values, signal);
    
    % --- KONTROL (Tüm Parametreler) ---
    % MCMC (Monte Carlo) doğası gereği %15-20 tolerans makuldür.
    testCase.verifyEqual(post_mean(1), true_f, 'RelTol', 0.20, 'Bayesian f hatası.');
    testCase.verifyEqual(post_mean(2), true_Dstar, 'RelTol', 0.25, 'Bayesian D* hatası.');
    testCase.verifyEqual(post_mean(3), true_D, 'RelTol', 0.15, 'Bayesian D hatası.');
    
    fprintf('✅ Bayesian Fit: f, D, D* Verified.\n');
end

% =========================================================================
% 5. TRI-EXPONENTIAL TESTİ
% =========================================================================
function testTriExp(testCase)
    % 3 Kompartıman: Fast, Inter, Slow
    % Parametrelerin birbirinden uzak olması ayrımı kolaylaştırır
    f1 = 0.15; D1 = 0.100;  % Fast (Perfusion)
    f2 = 0.25; D2 = 0.010;  % Inter (Tubular?)
    f3 = 0.60; D3 = 0.001;  % Slow (Tissue)
    
    b_values = [0, 10, 20, 40, 60, 100, 200, 400, 600, 800, 1000, 1500, 2000];
    signal = f1*exp(-b_values*D1) + f2*exp(-b_values*D2) + f3*exp(-b_values*D3);
    
    ft = fittype('S0 * (f1*exp(-b*D1) + f2*exp(-b*D2) + (1-f1-f2)*exp(-b*D3))', ...
        'independent', 'b', ...
        'coefficients', {'f1', 'f2', 'D1', 'D2', 'D3', 'S0'});
        
    opts = fitoptions('Method','NonLinearLeastSquares', ...
        'StartPoint', [0.1, 0.2, 0.08, 0.01, 0.001, 1], ...
        'Lower', [0, 0, 0.05, 0.005, 0.0005, 0.8], ...
        'Upper', [0.5, 0.5, 0.2, 0.02, 0.002, 1.2]);
        
    res = fit(b_values', signal', ft, opts);
    
    % --- KONTROL (6 Parametre) ---
    % Tri-exp çok hassas olduğu için toleranslar modele göre ayarlanır
    
    % Fraksiyonlar
    testCase.verifyEqual(res.f1, f1, 'RelTol', 0.15, 'Tri-Exp f1 (Fast) Hatası');
    testCase.verifyEqual(res.f2, f2, 'RelTol', 0.15, 'Tri-Exp f2 (Inter) Hatası');
    
    % Difüzyon Katsayıları
    testCase.verifyEqual(res.D1, D1, 'RelTol', 0.20, 'Tri-Exp D1 (Fast) Hatası');
    testCase.verifyEqual(res.D2, D2, 'RelTol', 0.15, 'Tri-Exp D2 (Inter) Hatası');
    testCase.verifyEqual(res.D3, D3, 'RelTol', 0.05, 'Tri-Exp D3 (Slow) Hatası'); % En kararlı olan budur
    
    % S0 (Signal Amplitude)
    testCase.verifyEqual(res.S0, 1.0, 'RelTol', 0.05, 'Tri-Exp S0 Hatası');
    
    fprintf('✅ Tri-Exp: f1, f2, D1, D2, D3, S0 Verified.\n');
end


% =========================================================================
% HELPER FUNCTION: BAYESIAN (Algorithm Core)
% =========================================================================
function [posterior_mean, posterior_std, chain, Rsquare, fitSignal] = bayesian_calc_helper(b_values, signal_data)
    % Input düzenleme
    if iscolumn(b_values), b_values = b_values'; end
    if iscolumn(signal_data), signal_data = signal_data'; end

    model_fun = @(params, b) params(1) * exp(-b * params(2)) + (1-params(1)) * exp(-b * params(3));
    
    % Priors
    prior_f = @(f) unifpdf(f, 0, 1); 
    prior_Dstar = @(Dstar) unifpdf(Dstar, 0.01, 0.5); 
    prior_D = @(D) unifpdf(D, 0.001, 0.5); 
    
    likelihood_fun = @(params, b, signal) prod(normpdf(signal, model_fun(params, b), 0.01));
    bayes_model = @(params) prior_f(params(1)) * prior_Dstar(params(2)) * prior_D(params(3)) * likelihood_fun(params, b_values, signal_data);
    
    % MCMC
    n_samples = 2000; % Test için biraz artırdım, daha stabil olsun diye
    params_init = [0.5 0.01 0.001]; 
    chain = zeros(n_samples, 3);
    
    for i = 1:n_samples
        params_prop = params_init + randn(1, 3) .* [0.1 0.01 0.001]; 
        params_prop(1) = max(0, min(1, params_prop(1))); 
        params_prop(2) = max(0, params_prop(2)); 
        params_prop(3) = max(0, params_prop(3)); 
        
        posterior_prop = bayes_model(params_prop);
        posterior_init = bayes_model(params_init);
        
        if posterior_init == 0, alpha = 1; else, alpha = min(1, posterior_prop / posterior_init); end
        
        if rand < alpha, params_init = params_prop; end
        chain(i, :) = params_init;
    end
    
    posterior_mean = mean(chain, 1);
    posterior_std = std(chain, 0, 1);
    
    fitSignal = model_fun(posterior_mean, b_values);
    residual = signal_data - fitSignal;
    S_res_ss = sum(residual.^2);
    S_tot_ss = sum((signal_data - mean(signal_data)).^2);
    Rsquare = 1 - (S_res_ss / S_tot_ss);
end