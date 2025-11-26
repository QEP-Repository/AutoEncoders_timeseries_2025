%% Causality AutoEncoder - CausAE
%
% Reference
%
% Riccardo Rossi et al. 
% On the Use of Autoencoders to Study the Dynamics and the Causality 
% Relations of Complex Systems with Applications to Nuclear Fusion 
% submitted to Computer Physics Communications

%% ========================================================================
%  Working Environment (CPU or GPU)
%  ========================================================================
%  Select whether the computation should be executed on CPU or GPU
%  If GPU is selected but not supported, it falls back to CPU.
% =========================================================================

Config.environemnt = "CPU"; % CPU, GPU

if Config.environemnt == "GPU"
    if canUseGPU
        gpu = gpuDevice();
        reset(gpu)
    
        gpu = gpuDevice();
        disp(gpu)
        wait(gpu)
    
        clear all; clc;
        disp("using GPU")
        Config.environemnt = "GPU";
    else
        clear; clc;
        Config.environemnt = "CPU";
        disp("cannot use GPU, CPU environemnt selected")
    end
else
    clear; clc;
    Config.environemnt = "CPU";
    disp("using CPU")
end

%% =========================================================================
%  Load Dataset
%  =========================================================================
%  Modify this section to load your specific dataset.
%  AR_C0.mat can be generated using Generate_AR.m
% =========================================================================

addpath 'Numerical Data'\

load("AR_C04.mat","t","x","y") % generate the dataset using Generate_AR.m (you have also HenonHenon system)

Data.t = t; % time
Data.X = x; % variable 1
Data.Y = y; % variable 2;

Data.labels = ["X";"Y"]; % define here the name, useful to define later cause and effect

clear t x y

Data.N = size(Data.(Data.labels(1)),2);

%% =========================================================================
%  CausAE Configuration Parameters
% =========================================================================

% Causal direction: driver → effect
CausAE.driver = "X";      % cause variable
CausAE.effect = "Y";      % effect variable

% Input configurations
CausAE.input_all = Data.labels;
CausAE.input_wod = Data.labels(Data.labels ~= CausAE.driver);  % without driver

% Time embedding for training
CausAE.time_window   = [1 1]; % time window for [driver effect]
CausAE.time_step     = [1 1]; % prediction step Δt for each variable
CausAE.buffer_step   = [1 1]; % stride during buffering

% Specify training regions (in normalized time [0,1])
CausAE.training_window = [0.0 0.3;
                          0.5 0.6;
                          0.7 1.0];

% Autoencoder architecture
CausAE.Layer_En = [20 20 20 20];
CausAE.Layer_Dec = flip(CausAE.Layer_En);
CausAE.CodeSize = 3;      % latent dimension

% Regularization
CausAE.sigma = 0.01;
CausAE.alpha = 1e-4;

% Training parameters
CausAE.max_epoch = 30;
CausAE.iter_per_epoch = 100;
CausAE.minibatch = 1000;
CausAE.LearningRate0 = 1e-3;
CausAE.DecayRate0 = 1e-4;
CausAE.MaxValidationChecks = 10;

% Ensembles configuration (for confidence estimation)
CausAE.Ensembles = 3;

Config.plot = 1;

addpath Functions\

%% =========================================================================
%  Time Window Buffering
% =========================================================================
%  Prepares sliding time windows for AE input matrices.
%  Produces both full and "without driver" versions.
% =========================================================================
Data = CausAE_Buffer(Data, CausAE);

%% =========================================================================
%  Train/Test Split
% =========================================================================
Data = CausAE_TrainingTestSet(Data, CausAE);

%% =========================================================================
%  Convert to dlarray for Deep Learning Processing
% =========================================================================
Train_Input_all = dlarray(Data.train_input_all, 'CB');
Train_Input_wod = dlarray(Data.train_input_wod, 'CB');
Train_Output    = dlarray(Data.train_output,    'CB');

Test_Input_all = dlarray(Data.test_input_all, 'CB');
Test_Input_wod = dlarray(Data.test_input_wod, 'CB');
Test_Output    = dlarray(Data.test_output,    'CB');

if Config.environemnt == "GPU"
    Train_Input_all = gpuArray(Train_Input_all);
    Train_Input_wod = gpuArray(Train_Input_wod);
    Train_Output    = gpuArray(Train_Output);
    Test_Input_all  = gpuArray(Test_Input_all);
    Test_Input_wod  = gpuArray(Test_Input_wod);
    Test_Output     = gpuArray(Test_Output);
end

%% =========================================================================
%  Training Ensemble
% =========================================================================
%  Two models are trained for each ensemble member:
%   1) Using full input (driver included)
%   2) Without driver (cause removed)
% =========================================================================

for e = 1:CausAE.Ensembles
    disp("==== Training Ensemble " + e + " of " + CausAE.Ensembles + " ====")

    Ensemble{e}.Model_All = CausAE_ModelTraining(Config,CausAE,...
        Train_Input_all,Train_Output,...
        Test_Input_all,Test_Output);

    Ensemble{e}.Model_wod = CausAE_ModelTraining(Config,CausAE,...
        Train_Input_wod,Train_Output,...
        Test_Input_wod,Test_Output);
end

%% =========================================================================
%  Error & Causality Analysis
% =========================================================================
Results = CausAE_ErrorAnalysis_Ensemble(Data, CausAE, Ensemble);

disp("Analysis completed successfully.")