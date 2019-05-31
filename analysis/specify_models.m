% Specify all models used in main.m:
%
% Hypotetical Models
%  - model1
%  - model2
%  - model3
%  - model4
%  - model5
%  - model6
%  - model7
%  - model8
%
%  "True" models
%  - modelControl
%  - modelSchizophrenia1
%  - modelSchizophrenia2

% WARNING: Some labeling comments can be outdated :/


b = 0;

%% Model 1

clean("P");

P.A = [
%    from
%   I I F I
%   T N P F
%   G S   G   to
    0 b b 0        % ITG (object recognition)
    1 0 0 b;   % INS (audio/vision integration)
    1 0 0 b; % FP (monitor current action)
    0 1 1 0;   % IFG  (impulse control)
    ];

P.B = [
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    0 -1 0 0;
    ];

P.C = [
    0.1 0;
    0 0;
    0 0;
    0 0
    ];

P.D1 = [
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    ];

P.D2 = [
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    ];

P.kappa = 0;
P.gamma = 0;
P.tau = 0;
P.alpha = 0;
P.E0 = 0;

save_model("model1", P);

%% Model 2
P.A = [
%    from
%   I I F I
%   T N P F
%   G S   G   to
    0 b b 0        % ITG (object recognition)
    1 0 0 b;   % INS (audio/vision integration)
    1 0 0 b; % FP (monitor current action)
    0 1 1 0;   % IFG  (impulse control)
    ];

P.B = [
    0 0 0 0;
    -0.1 0 0 0;
    0 0 0 0;
    0 -0.1 0 0;
    ];

P.C = [
    0.1 0;
    0 0;
    0 0;
    0 0
    ];

P.D1 = [
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    ];

P.D2 = [
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    ];

P.kappa = 0;
P.gamma = 0;
P.tau = 0;
P.alpha = 0;
P.E0 = 0;

save_model("model2", P);

%% Model 3

P.A = [
%    from
%   I I F I
%   T N P F
%   G S   G   to
    0 -.5 -.5 0        % ITG (object recognition)
    1 0 0 -.5;   % INS (audio/vision integration)
    1 0 0 -.5; % FP (monitor current action)
    0 1 1 0;   % IFG  (impulse control)
    ];

P.B = [
    0 0 0 0;
    0 0 0 0;
    -1 0 0 0;
    0 -1 0 0;
    ];

P.C = [
    0.1 0;
    0 0;
    0 0;
    0 0
    ];

P.D1 = [
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    ];

P.D2 = [
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    ];

P.kappa = 0;
P.gamma = 0;
P.tau = 0;
P.alpha = 0;
P.E0 = 0;

save_model("model3", P);

%% Model 4
P.A = [
%    from
%   I I F I
%   T N P F
%   G S   G   to
    0 -.5 -.5 -.5        % ITG (object recognition)
    1 0 0 -.5;   % INS (audio/vision integration)
    1 0 0 -.5; % FP (monitor current action)
    1 1 1 0;   % IFG  (impulse control)
    ];

P.B = [
    0 0 0 0;
    -1 0 0 0;
    0 0 0 0;
    -1 0 0 0;
    ];

P.C = [
    0.1 0;
    0 0;
    0 0;
    0 0
    ];

P.D1 = [
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    ];

P.D2 = [
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    ];

P.kappa = 0;
P.gamma = 0;
P.tau = 0;
P.alpha = 0;
P.E0 = 0;

save_model("model4", P);

%% Model 5
P.A = [
%    from
%   I I F I
%   T N P F
%   G S   G   to
    0 -.5 -.5 0        % ITG (object recognition)
    1 0 0 -.5;   % INS (audio/vision integration)
    1 0 0 -.5; % FP (monitor current action)
    0 1 1 0;   % IFG  (impulse control)
    ];

P.B = [
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    0 -1 0 0;
    ];

P.C = [
    0.1 0;
    0 0;
    0 0;
    0 0
    ];

P.D1 = [
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    0 0 1 0;
    ];

P.D2 = [
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    ];

P.kappa = 0;
P.gamma = 0;
P.tau = 0;
P.alpha = 0;
P.E0 = 0;

save_model("model5", P);

%% Model 6
P.A = [
%    from
%   I I F I
%   T N P F
%   G S   G   to
    0 -.5 -.5 0        % ITG (object recognition)
    1 0 0 -.5;   % INS (audio/vision integration)
    1 0 0 -.5; % FP (monitor current action)
    0 1 1 0;   % IFG  (impulse control)
    ];

P.B = [
    0 0 0 0;
    -1 0 0 0;
    0 0 0 0;
    0 0 0 0;
    ];

P.C = [
    0.1 0;
    0 0;
    0 0;
    0 0
    ];

P.D1 = [
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    ];

P.D2 = [
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    0 1 0 0;
    ];

P.kappa = 0;
P.gamma = 0;
P.tau = 0;
P.alpha = 0;
P.E0 = 0;

save_model("model6", P);

%% Model 7

P.A = [
%    from
%   I I F I
%   T N P F
%   G S   G   to
    0 -.5 -.5 0        % ITG (object recognition)
    1 0 0 -.5;   % INS (audio/vision integration)
    1 0 0 -.5; % FP (monitor current action)
    0 1 1 0;   % IFG  (impulse control)
    ];

P.B = [
    0 0 0 0;
    0 0 0 0;
    -1 0 0 0;
    0 0 0 0;
    ];

P.C = [
    0.1 0;
    0 0;
    0 0;
    0 0
    ];

P.D1 = [
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    0 0 1 0;
    ];

P.D2 = [
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    0 1 0 0;
    ];

P.kappa = 0;
P.gamma = 0;
P.tau = 0;
P.alpha = 0;
P.E0 = 0;

save_model("model7", P);

%% Model 8
P.A = [
%    from
%   I I F I
%   T N P F
%   G S   G   to
    0 -.5 -.5 -.5        % ITG (object recognition)
    1 0 0 -.5;   % INS (audio/vision integration)
    1 0 0 -.5; % FP (monitor current action)
    1 1 1 0;   % IFG  (impulse control)
    ];

P.B = [
    0 0 0 0;
    -1 0 0 0;
    0 0 0 0;
    -1 0 0 0;
    ];

P.C = [
    0.1 0;
    0 0;
    0 0;
    0 0
    ];

P.D1 = [
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    1 0 0 0;
    ];

P.D2 = [
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    1 0 0 0;
    ];
P.kappa = 0;
P.gamma = 0;
P.tau = 0;
P.alpha = 0;
P.E0 = 0;

save_model("model8", P);

%% "True" Model Control
P.A = [
%    from
%   I I F I
%   T N P F
%   G S   G   to
    0 -.3 -.3 -.3        % ITG (object recognition)
    .4 0 0 -.3;   % INS (audio/vision integration)
    .5 0 0 -.3; % FP (monitor current action)
    -.2 -.4 .8 0;   % IFG  (impulse control)
    ];

P.B = [
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    .6 0 0 0;
    ];

P.C = [
    0.1 0;
    0 0;
    0 0;
    0 0
    ];

P.D1 = [
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    .6 0 0 0;
    ];

P.D2 = [
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    .6 0 0 0;
    ];
P.kappa = 0;
P.gamma = 0;
P.tau = 0;
P.alpha = 0;
P.E0 = 0;

save_model("modelControl", P);

%% "True" Model Schizoprhenia 1
P.A = [
%    from
%   I I F I
%   T N P F
%   G S   G   to
    0 -.3 -.3 -.3        % ITG (object recognition)
    .4 0 0 -.3;   % INS (audio/vision integration)
    .5 0 0 -.3; % FP (monitor current action)
    -.4 -.4 .1 0;   % IFG  (impulse control)
    ];

P.B = [
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    ];

P.C = [
    0.1 0;
    0 0;
    0 0;
    0 0
    ];

P.D1 = [
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    .6 0 0 0;
    ];

P.D2 = [
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    .6 0 0 0;
    ];
P.kappa = 0;
P.gamma = 0;
P.tau = 0;
P.alpha = 0;
P.E0 = 0;

save_model("modelSchizophrenia1", P);

%% "True" Model Schizoprhenia
P.A = [
%    from
%   I I F I
%   T N P F
%   G S   G   to
    0 -.3 -.3 -.3        % ITG (object recognition)
    .4 0 0 -.3;   % INS (audio/vision integration)
    .5 0 0 -.3; % FP (monitor current action)
    -.2 -.6 .1 0;   % IFG  (impulse control)
    ];

P.B = [
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    ];

P.C = [
    0.1 0;
    0 0;
    0 0;
    0 0
    ];

P.D1 = [
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    .6 0 0 0;
    ];

P.D2 = [
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    .6 0 0 0;
    ];
P.kappa = 0;
P.gamma = 0;
P.tau = 0;
P.alpha = 0;
P.E0 = 0;

save_model("modelSchizophrenia2", P);