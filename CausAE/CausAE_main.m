% Causality AutoEncoder Ensemble

gpu = gpuDevice();
reset(gpu)

gpu = gpuDevice();
disp(gpu)
wait(gpu)

clear all; clc;

%% here you need to upload the dataset you want to analyse

addpath 'Numerical Data'\

load("AR_C04.mat","t","x","y") % generate the dataset using Generate_AR.m (you have also HenonHenon system)

Data.t = t; % time
Data.X = x; % variable 1
Data.Y = y; % variable 2;

Data.labels = ["X";"Y"]; % define here the name, useful to define later cause and effect

clear t x y

Data.N = size(Data.(Data.labels(1)),2);

%% CausAE configuration

CausAE.driver = "X"; % define the driver
CausAE.effect = "Y"; % define the effect (caused variable)

CausAE.input_all = Data.labels;
CausAE.input_wod = Data.labels(Data.labels~=CausAE.driver); 

CausAE.time_window = [1 1];   % time window [x y]
CausAE.time_step = [1 1];     % Deltat prediction [x y]
CausAE.buffer_step = [1 1];     % step of the buffer (for training) [x y]

CausAE.training_window = [0 0.3;0.5 0.6;0.7 1];     % Define the time window for training (results evaluated on test)

CausAE.Layer_En = [20 20 20 20];
CausAE.Layer_Dec = flip(CausAE.Layer_En);
CausAE.CodeSize = 3;
CausAE.sigma = 0.01;
CausAE.alpha = 1e-4;

CausAE.MaxValidationChecks = 10;
CausAE.max_epoch = 30;
CausAE.iter_per_epoch = 100;
CausAE.minibatch = 1000;

CausAE.LearningRate0 = 1e-3;
CausAE.DecayRate0 = 1e-4;

CausAE.Ensembles = 3;

addpath Functions\

%% Prepare Time Windows

Data = CausAE_Buffer(Data,CausAE);

%% Prepare Training and Test Set

Data = CausAE_TrainingTestSet(Data,CausAE);

%% dlarray Data

Train_Input_all = dlarray(Data.train_input_all,'CB');
Train_Input_wod = dlarray(Data.train_input_wod,'CB');
Train_Output = dlarray(Data.train_output,'CB');

Test_Input_all = dlarray(Data.test_input_all,'CB');
Test_Input_wod = dlarray(Data.test_input_wod,'CB');
Test_Output = dlarray(Data.test_output,'CB');

%%

for e1 = 1 : CausAE.Ensembles

    Ensemble{e1}.Model_All = CausAE_ModelTraining(Data,CausAE,...
                            Train_Input_all,Train_Output,...
                            Test_Input_all,Test_Output);
    
    Ensemble{e1}.Model_wod = CausAE_ModelTraining(Data,CausAE,...
                            Train_Input_wod,Train_Output,...
                            Test_Input_wod,Test_Output);

end

%% Analyse results

Results = CausAE_ErrorAnalysis_Ensemble(Data,CausAE,Ensemble);
