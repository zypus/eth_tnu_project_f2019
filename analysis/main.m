%% Manual Experimentation, model loading and simulations

u= zeros(2, 800);

for i=1:800
    if mod(i,70) == 0 && i < 700
        u(1,i) = 5;
    end
    if i >= 300 && i <= 600
        u(2,i) = 1;
    end
end


%sub = simulate_new_data_for_subject("sub-10704", "Control", 0.1); 
sub = simulate_new_data_for_subject("sub-50004", "Schizophrenia1", 0.1); 
%sub = load_subject("sub-50004");

U.u = sub.u;
U.dt = 0.1;
U.subsample = 20;

test = 0;
simulate = 1;
model = 7;

if test == 1
    P.A = [0, 0; 1, 0];
    P.B = [0, 0; -0.5, 0];
    P.C = [1, 0; 0, 0];
    P.D = [0, 0; 0, 0];
    P.kappa = 0;
    P.gamma = 0;
    P.tau = 0;
    P.alpha = 0;
    P.E0 = 0;
else
    P = load_model("model" + model);
end

mask = structure_parameters(flatten_parameters(P) ~= 0, P);

% mask.A = [0, 0; 1, 0];
% mask.B = [0, 0; 1, 0];
% mask.C = [1, 0; 0, 0];
% mask.kappa = 0;
% mask.gamma = 0;
% mask.tau = 0;
% mask.alpha = 0;
% mask.E0 = 0;

P_hrf.kappa = 0.64;
P_hrf.gamma = 0.32;
P_hrf.tau = 2;
P_hrf.alpha = 0.32;
P_hrf.E0 = 0.4;

if test == 1
    x0 = [0;0];
else
    x0 = zeros(size(sub.y,1),1);
end

h0 = [0, 1, 1, 1]';

[y, h, x] = euler_integrate_dcm(U, P, P_hrf, x0, h0);

if simulate == 1
    stdY = mean(std(y'));
    y_head = y + stdY * randn(size(y));

    y_head = y_head(:,1:U.subsample:end);
else
    stdY = mean(std(sub.y'));

    y_head = sub.y;
end

run_custom = 0;

if run_custom == 1
    fun = @(p) -log_likelihood(y_head, U, merge_parameters(P, mask, p), P_hrf, x0, h0, stdY);
    fun2 = @(p) -log_joint_distribution(y_head, U, merge_parameters(P, mask, p), P_hrf, x0, h0, stdY, mask, 1);

    p0 = zeros(sum(flatten_parameters(mask)),1);



    pMAP = fminsearch(fun2, p0)

    P2 = merge_parameters(P, mask, pMAP);

    P2.A - P.A
    P2.B - P.B
    P2.C - P.C
end

%% Plot Actual BOLD signal

subplot(2,1,1);
t = (1:size(U.u,2))*U.dt;
plot(t, U.u./[1;-1]);
ylim([-1.4 1.4]);
title("Input ($u_2$ has been scaled by -1 for illustration only)");
xlabel("time");
ylabel("amplitude");
legend("$u_2$ (ie. Go-Signal)", "$u_1$ (ie. Stop-Signal)");

subplot(2,1,2);
plot(t(1:U.subsample:end), sub.y, "-s", "MarkerSize", 2);
title("BOLD Signal (TR=2sec)");
xlabel("time");
ylabel("amplitude");
legend("ITG", "INS", "FP", "IFG");

shg()
saveas(gcf, "../figures/BOLD.png");

%% Plot Simulated Data

subplot(4,1,1);
t = (1:size(U.u,2))*U.dt;
plot(t, U.u./[1;-1]);
ylim([-1.4 1.4]);
title("Input ($u_2$ has been scaled by -1 for illustration only)");
xlabel("time");
ylabel("amplitude");
legend("$u_2$ (ie. Go-Signal)", "$u_1$ (ie. Stop-Signal)");

subplot(4,1,2);
plot(t, x);
title("Neural State");
xlabel("time");
ylabel("activity");
legend("ITG", "INS", "FP", "IFG");

subplot(4,1,3);
plot(t, y);
title("BOLD Signal");
xlabel("time");
ylabel("amplitude");
legend("ITG", "INS", "FP", "IFG");

subplot(4,1,4);
plot(t(1:U.subsample:end), y_head, "-s", "MarkerSize", 2);
title("Measured BOLD Signal (noise, sampling frequency $f=\frac{1}{20}$)");
xlabel("time");
ylabel("amplitude");
legend("ITG", "INS", "FP", "IFG");

shg()
saveas(gcf, "../figures/Simulation.png");


%% Negative Free Energy

%x0 = y_head(:,1);

clear("M", "Y");

M.IS = @(p,M,U) euler_integrate_dcm_y(U, merge_parameters(P, mask, p), P_hrf, x0, h0)';
M.pE = zeros(sum(flatten_parameters(mask)),1);
M.pC = eye(sum(flatten_parameters(mask)));
%M.hE = 1;
%M.hC = 1000;
%M.P = P;
Y.y = y_head';
Y.dt = 2;
%if simulate == 0
%    Y.X0 = sub.x0;
%end
%Y.Q = stdY;

[Ep, ~, ~, F] = spm_nlsi_GN(M,U,Y)

P3 = merge_parameters(P, mask, Ep);


%% MCMC (requires MatlabMutliNest [https://github.com/mattpitkin/matlabmultinest])

%addpath("matlabmultinest/general");
%addpath("matlabmultinest/src");

% global verbose;
% verbose = 1;
% global DEBUG;
% DEBUG = 0;
% 
% sigma = mean(std(y_head'));
% 
% fmodel = @(x, pn, pv) euler_integrate_dcm_y(U, convert_to_parameters(pn, pv, template), P_hrf, x0, h0)';
% floglike = @(y_head, model, pn, pv) mcmc_log_likelihood(y_head, model, pn, pv, sigma);
% 
% param_index = 1;
% flat_mask = flatten_parameters(mask);
% prior = {};
% for i=1:size(flat_mask,1)
%     if flat_mask(i) == 1
%         prior{param_index,1} = char(string(i));
%         prior{param_index,2} = 'gaussian';
%         prior{param_index,3} = 0;
%         prior{param_index,4} = 1;
%         prior{param_index,5} = '';
%         param_index = param_index + 1;
%     end
% end
% 
% [logZ, nest_samples, post_samples] = nested_sampler(y_head, 500, 0.1, floglike, fmodel, prior, {}, 'Nmcmc', 0, 'totsamples', 200);
% 
% P3 = merge_parameters(P, mask, mean(post_samples(:,1:size(prior,1))));

%% Plot Actual vs Estimated

[y2, h2, x2] = euler_integrate_dcm(U, P3, P_hrf, x0, h0);

subplot(6,1,1);
t = (1:size(U.u,2))*U.dt;
plot(t, U.u);

subplot(6,1,2);
plot(t, x);

subplot(6,1,3);
plot(t, y);

%subplot(6,1,4);
%plot(t, resample(y_head',U.subsample,1)');

subplot(6,1,4);
plot(t(1:U.subsample:end), y_head, "-s", "MarkerSize", 4);

subplot(6,1,5);
plot(t, x2);

subplot(6,1,6);
plot(t, y2);

shg()

%% All subjects all models
% run estimation

% sim = 0 -> actual data
% sim = 1 -> simulated data
sim = 0;
few_subjects = 1;

if sim == 1 && exist("../data/matlab/sim_dcms", "dir") ~= 7
    mkdir("../data/matlab/sim_dcms")
end

if sim == 0 && exist("../data/matlab/dcms", "dir") ~= 7
    mkdir("../data/matlab/dcms")
end

if few_subjects == 1
    subjects = get_subject_list_sample(60, 30);
else
    subjects = get_subject_list();
end

Fmat = zeros(size(subjects, 1), 8);

for s=1:size(subjects,1)
    try
        subject_id = subjects(s);
        
        if sim == 1
            if startsWith(subject_id, "sub-1")
                sub = simulate_new_data_for_subject(subject_id, "Control", 0.1);
            else
                chars = char(subject_id);
                if mod(str2double(chars(9)), 2) == 0
                    sub = simulate_new_data_for_subject(subject_id, "Schizophrenia1", 0.1);
                else
                    sub = simulate_new_data_for_subject(subject_id, "Schizophrenia2", 0.1);
                end
            end
        else
            sub = load_subject(subject_id);
        end
     
        fits = cell(8,1);
        parfor m=1:8
            try
                if sim == 1
                    file = "../data/matlab/sim_dcms/dcm_" + subject_id + "_model" + m + ".mat";
                else
                    file = "../data/matlab/dcms/dcm_" + subject_id + "_model" + m + ".mat";
                end
                
                if exist(file, "file") ~= 0
                   loaded = load(file);
                   fits{m} = loaded.fit;
                else
                   fits{m} = fit_model_to_subject(m, sub);
                end

                Fmat(s,m) = fits{m}.F;
            catch error
               warning("Subject index " + s + " failed with model" + m + ": " + error.message);
            end
        end
        for m=1:8
            if sim == 1
                file = "../data/matlab/sim_dcms/dcm_" + subject_id + "_model" + m + ".mat";
            else
                file = "../data/matlab/dcms/dcm_" + subject_id + "_model" + m + ".mat";
            end
            if exist(file, "file") == 0
                fit = fits{m};
                save(file, "fit");
            end
        end
    catch error
        warning("Subject index " + s + " failed: " + error.message);
    end
end

if sim == 1
    save("../data/matlab/sim_dcms/Fmat.mat", "Fmat");
else
    save("../data/matlab/dcms/Fmat.mat", "Fmat");
end

%% Load data Fmat

% sim = 0 -> actual data
% sim = 1 -> simulated data
sim = 1;

if sim == 1
    load("../data/matlab/sim_dcms/Fmat.mat");
else
    load("../data/matlab/dcms/Fmat.mat");
end

%% FFX

model_count = size(Fmat, 2);

log_bayes_factors = zeros(model_count, model_count, size(Fmat,1));
posterior_model_prob = zeros(size(Fmat));

for s=1:size(Fmat,1)
    log_bayes_factors(:,:,s) = repmat(Fmat(s,:), model_count, 1)-Fmat(s,:)'; 
    posterior_model_prob(s,:) = sum(exp(log_bayes_factors(:,:,s))).^-1;
end

log_gbf = sum(log_bayes_factors,3);

log_abf = -size(Fmat, 1) * log_gbf;

set(groot,'defaulttextinterpreter','latex');  
set(groot, 'defaultAxesTickLabelInterpreter','latex');  
set(groot, 'defaultLegendInterpreter','latex'); 

subplot(2,1,1);
hm = heatmap(log_gbf, "Title", "log GBF(b,a)");
xlabel("model a");
ylabel("model b");

subplot(2,1,2);
bar(mean(posterior_model_prob));
hold on;
errorbar(mean(posterior_model_prob), std(posterior_model_prob));
hold off;
title("Posterior Model Probabilites");
xlabel("model");
ylabel("posterior model probability");
shg();
saveas(gcf, "../figures/FFX" + sim + ".png");

family.infer = 'FFX';
family.partition = [1 2 3 4 5 6 7 8];
family.names = {"1", "2", "3", "4", "5", "6", "7", "8"};

[family, model] = spm_compare_families(Fmat, family);

subplot(1,1,1);
bar(family.post);
title("Posterior Model Probabilites");
xlabel("model");
ylabel("posterior model probability");
shg();
saveas(gcf, "../figures/FFX2" + sim + ".png");


%% RFX

[alpha, exp_r, xp, pxp, bor] = spm_BMS(Fmat,1e6,1,0,1);

subplot(2,2,[1 3]);
bar(1:8, exp_r);
xlabel("model");
ylabel("expected probability");

subplot(2,2,2);
bar(1:8, xp);
xlabel("model");
ylabel("exceedance probability");

subplot(2,2,4)
bar(1:8,pxp(:)')
ylabel('protected exceedance probability')
xlabel('Models')
title(sprintf('Prob of Equal Model Frequencies (BOR) = %1.2f',bor));
shg()

saveas(gcf, "../figures/RFX" + sim + ".png");

%% Family FFX

clear("family");

family.infer = 'FFX';
family.partition = [1 1 1 1 2 2 2 2];
family.names = {"bilinear", "nonlinear"};

[family, model] = spm_compare_families(Fmat, family);

subplot(1,1,1);
bar(categorical({'linear', 'nonlinear'}), family.post);
title("Posterior Family Probabilites");
xlabel("families");
ylabel("posterior family probability");
shg();
saveas(gcf, "../figures/FamilyFFX" + sim + ".png");

%% Family RFX

clear("family");

family.infer = 'RFX';
family.partition = [1 1 1 1 2 2 2 2];
family.names = {"bilinear", "nonlinear"};

[family, model] = spm_compare_families(Fmat, family);

subplot(2,1,1);
bar(categorical({'linear', 'nonlinear'}), family.exp_r);
xlabel("families");
ylabel("family expected probability");

subplot(2,1,2);
bar(categorical({'linear', 'nonlinear'}), family.xp);
xlabel("families");
ylabel("family exceedance probability");
shg();

saveas(gcf, "../figures/FamilyRFX" + sim + ".png");

%% PCA and clustering


if sim == 0
    best_model = 5;
    model_param_names = {'ITG=-INS','ITG=-FP','ITG-=INS','INS=-IFG','ITG-=FP','FP=-IFG','INS-=IFG','FP-=IFG','stop(INS=-IFG)', 'go(ITG)', 'INS(FP=-IFG)'};
else
    best_model = 7;
    model_param_names = {'ITG=-INS','ITG=-FP','ITG-=INS','INS=-IFG','ITG-=FP','FP=-IFG','INS-=IFG','FP-=IFG','stop(ITG=-FP)', 'go(ITG)', 'INS(FP=-IFG)', 'FP(INS->IFG)'};
end

if sim == 0
    subjects = get_subject_list_sample(60,30);
else
    subjects = get_subject_list();
end

model = load_model("model" + best_model);
mask = flatten_parameters(model) ~= 0;

clear("parameters_per_subject");
for s=1:size(subjects,1)
    subject_id = subjects(s);
    if sim == 1
        file = "../data/matlab/sim_dcms/dcm_" + subject_id + "_model" + best_model + ".mat";
    else
        file = "../data/matlab/dcms/dcm_" + subject_id + "_model" + best_model + ".mat";
    end
    dcm = load(file);
    
    idx = 1;
    flat_params = flatten_parameters(dcm.fit.P);
    for i=1:size(mask, 1)
        if mask(i) == 1
            parameters_per_subject(s, idx) = flat_params(i);
            idx = idx + 1;
        end
    end
end

[coeff, scores] = pca(parameters_per_subject);

figure(1);
subplot(1,1,1);
biplot(coeff(:,1:2),'scores',scores(:,1:2),'varlabels',model_param_names);
saveas(gcf, "../figures/PCA" + sim + ".png");
shg()

subjects_as_char = char(subjects);
if sim == 0
    label = startsWith(subjects, "sub-5");
    types = {'Control', 'Schizophrenia'};
else
    label = startsWith(subjects, "sub-5") + (startsWith(subjects, "sub-5").*(mod(subjects_as_char(:,9), 2) == 1));
    types = {'Control', 'Schizophrenia1', 'Schizophrenia2'};
end

figure(2);
subplot(3,2,[1 2]);
gscatter(scores(:,1), scores(:,2),label,'brm', '.**');
title("Subjects by $pca_1$ and $pca_2$");
xlabel("$pca_1$");
ylabel("$pca_2$");
legend(types);
%legend('Control', 'Schizophrenia1', 'Schizophrenia2');

rng(1);
[idx,C] = kmeans(scores, 2);

subplot(3,2,3);
gscatter(scores(:,1), scores(:,2), idx, 'cmgb');
title("Kmeans(K=2)");
ylabel("$pca_2$");
legend('Cluster 1', 'Cluster 2');

regularization = 0.01;
rng('default');  % For reproducibility
fitter = @(X,K) cluster(fitgmdist(X,K,'RegularizationValue',regularization), X);
eva = evalclusters(scores,fitter,'CalinskiHarabasz','KList', 1:5);

subplot(3,2,4);
bar(eva.CriterionValues);
title("Optimal number of clusters");
ylabel("Calinski Harabasz");


rng(1);
gmfit = fitgmdist(scores,2,'RegularizationValue',regularization);
clusterX = cluster(gmfit,scores);

subplot(3,2,5);
gscatter(scores(:,1), scores(:,2), clusterX, 'cmg');
title("Gaussian Mixture Model based clustering(K=2)");
xlabel("$pca_1$");
ylabel("$pca_2$");
legend('Cluster 1', 'Cluster 2');
shg();

rng(1);
gmfit = fitgmdist(scores,3,'RegularizationValue',regularization);
clusterX2 = cluster(gmfit,scores);

subplot(3,2,6);
gscatter(scores(:,1), scores(:,2), clusterX2, 'cmg');
title("Gaussian Mixture Model based clustering(K=3)");
xlabel("$pca_1$");
ylabel("$pca_2$");
legend('Cluster 1', 'Cluster 2', 'Cluster 3');
shg();

saveas(gcf, "../figures/Clustering" + sim + ".png");

positive = startsWith(subjects, "sub-5");

confMat = confusionmat(positive, clusterX==2);

for i =1:size(confMat,1)
    recall(i)=confMat(i,i)/sum(confMat(i,:));
    precision(i)=confMat(i,i)/sum(confMat(:,i));
end
Recall=sum(recall)/size(confMat,1)
Precision=sum(precision)/size(confMat,1)
F_score=2*Recall*Precision/(Precision+Recall)

figure(3);
subplot(1,1,1);
heatmap(confMat);
title("Confusion Matrix")
xlabel("Predicted Class");
ylabel("Actual Class");
shg();

saveas(gcf, "../figures/Confusion" + sim + ".png");


%% Average Models

avg_model1 = mean(parameters_per_subject(clusterX == 1,:));
std_model1 = std(parameters_per_subject(clusterX == 1,:));

avg_model2 = mean(parameters_per_subject(clusterX == 2,:));
std_model2 = std(parameters_per_subject(clusterX == 2,:));

diff_model = merge_parameters(model, structure_parameters(mask, model), avg_model1 - avg_model2);

diff_model.A
diff_model.B
diff_model.C
diff_model.D1
diff_model.D2


bar(categorical(model_param_names), avg_model1 - avg_model2)
hold on;
errorbar(categorical(model_param_names), avg_model1 - avg_model2, std_model1 + std_model2, '.');
hold off;
title("Comparing average models (Cluster1 - Cluster2)")
xlabel("Parameter")
ylabel("Difference")
shg()

saveas(gcf, "../figures/AverageModels" + sim + ".png");