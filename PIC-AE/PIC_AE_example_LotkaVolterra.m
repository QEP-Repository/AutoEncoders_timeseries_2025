%% Physics-Informed Code AutoEncoder - PICAE
%
%  Script developed for Lotka-Volterra example
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

%% ========================================================================
%  Add Required Paths
% =========================================================================

addpath Dataset\
addpath Functions\

disp("paths added")

%% ========================================================================
%  Load Data
%  ========================================================================
%  Example with Lotka–Volterra simulated dataset
%  For experimental cases, replace this block with dataset import
% =========================================================================

load("LotkaVolterra.mat")

% Construct proxy variables
p = (LV.x+LV.y)/2;
q = LV.x./(1+LV.y);

% Observable system variables (optional: depending on configuration)
x = LV.x;
y = LV.y;

% Add Gaussian noise to data
I_noise = 0.1; % percentage noise level
p = normrnd(p,I_noise.*std(p));
q = normrnd(q,I_noise.*std(q));
x = normrnd(x,I_noise.*std(x));
y = normrnd(y,I_noise.*std(y));

Config.I_noise = I_noise; clear I_noise

%% ========================================================================
%  Prepare Data for PIC-AE
% =========================================================================

% Proxies
Data.prox{1} = p;
Data.prox{2} = q;

% Observables — sampled as sparse measurements
Data.Ind_obs = 500:200:1e4;
Data.xobs = x(Data.Ind_obs);
Data.yobs = y(Data.Ind_obs);

clear x y p q

%% ========================================================================
%  Model Parameters (only used if SelfModelling = 0), dt is needed
% =========================================================================

Model.dt = dlarray(LV.dt,'CB');
Model.a = dlarray(LV.a,'CB');
Model.b = dlarray(LV.b,'CB');
Model.c = dlarray(LV.c,'CB');
Model.d = dlarray(LV.d,'CB');

%% ========================================================================
%  PIC-AE Configuration
% =========================================================================

PICAE.SelfModelling = 1; % If 0: use Model parameters | If 1: learn them
PICAE.UseObservale = 1; % If 1: include observable constraints

PICAE.Buffer_Size = 5;
PICAE.Buffer_Res = 1;

PICAE.step = 1; 

PICAE.iter_per_epoch = 100;
PICAE.max_epoch = 100;

PICAE.MiniBatch = 1000;
PICAE.PhysicsBatch = 1000;
PICAE.LearningRate0 = 1e-3;
PICAE.DecayRate0 = 0;

% architecture
PICAE.Layer_En = [20 20 20 20 20 20 20];
PICAE.Layer_Dec = flip(PICAE.Layer_En);
PICAE.CodeSize = 2;

% Weighiting scheme
PICAE.alpha_range = [1 inf];
PICAE.alpha = 1;
PICAE.alpha_adaptive = 1;
if PICAE.UseObservale == 1
    PICAE.MSE_target = 4.75*Config.I_noise.^2.*(std(Data.prox{1}).^2 + std(Data.prox{2}).^2)/2;
else 
    PICAE.MSE_target = 2*Config.I_noise.^2.*(std(Data.prox{1}).^2 + std(Data.prox{2}).^2)/2;
end

% plot options
Config.plot = 1;
Config.plot_rate = 1; % x means that it plots every x epochs
Config.plot_ind = 1:3000;

%% ========================================================================
%  Buffering: Convert Time Series into Proxies Window
% =========================================================================

Data_Prox_buff = [];

for i = 1 : length(Data.prox)

    Data_Prox_buff = [Data_Prox_buff; buffer(Data.prox{i}, ...
                    PICAE.Buffer_Size, ...
                    PICAE.Buffer_Size-PICAE.Buffer_Res)];

end

if ~isempty(Data.Ind_obs)
    Data_Obs_buff = Data_Prox_buff(:,Data.Ind_obs);
else
    Data_Obs_buff = "none";
end

% Remove initial zero-padding due to buffering
Data_Prox_buff(:,1:PICAE.Buffer_Size) = [];

Data.Input_proxies = Data_Prox_buff;
Data.Input_observable = Data_Obs_buff;

clear i Data_Obs_buff Data_Prox_buff 

%% ========================================================================
%  Convert to Deep Learning Arrays (dlarray)
% =========================================================================

% P = Proxies
inds = 1:size(Data.Input_proxies,2);
dlP_0 = dlarray(Data.Input_proxies(:,inds(1)+PICAE.step:inds(end)),'CB'); % no delay
dlP_1 = dlarray(Data.Input_proxies(:,inds(1):inds(end)-PICAE.step),'CB'); % one delay
if Config.environemnt =="GPU"
    dlP_0 = gpuArray(dlP_0);
    dlP_1 = gpuArray(dlP_1);
end

% Pc = Proxies for code constrain at known observables /
% Oc = observable for code
if ~isempty(Data.Ind_obs) && PICAE.UseObservale==1
    dlPc = dlarray(Data.Input_observable,'CB');
    dlOc = dlarray([Data.xobs; Data.yobs],'CB');
    if Config.environemnt =="GPU"
        dlPc = gpuArray(dlPc);
        dlOc = gpuArray(dlOc);
    end
end

%% ========================================================================
%  Initialise the PIC-AE Architecture and Optimiser
% =========================================================================

[~,~,parameters] = AE_network_LV(dlP_0(:,1:10),dlP_0(:,1:10),0,[],PICAE,0);
[dlY0,Code] = AE_network_LV(dlP_0(:,1:10),[],1,parameters,PICAE,0);

averageGrad = [];
averageSqGrad = [];

clear dlY0 Code

%% ========================================================================
%  Select Correct Gradient Function (depending on configuration)
% =========================================================================

if PICAE.SelfModelling == 0
    accfun = dlaccelerate(@ModelGradient_LV);
elseif PICAE.SelfModelling == 1 && PICAE.UseObservale == 0
    accfun = dlaccelerate(@ModelGradient_LV_SM);
else
    accfun = dlaccelerate(@ModelGradient_LV_SM_withObservable);
end

%% ========================================================================
%  Plot Initialisation
% =========================================================================

if Config.plot == 1
   figure(1)
   clf
   subplot(2,2,1)
   yline(PICAE.MSE_target)
   grid on
   grid minor
   xlabel("epoch[arb.units]")
   ylabel("Loss [arb.units]")
   hold on
   set(gca,'YScale','log')
end

%% ========================================================================
%  Training Loop
% =========================================================================

alpha = PICAE.alpha;

iteration = 0;

for epoch = 1 : PICAE.max_epoch

    for i = 1 : PICAE.iter_per_epoch

        % iteration update
        iteration = iteration + 1;

        % mini batch extraction
        ind_batch = randsample(size(dlP_0,2),PICAE.MiniBatch);

        dlP_0_now = dlP_0(:,ind_batch);
        dlP_1_now = dlP_1(:,ind_batch);

        % model gradient
        if PICAE.SelfModelling == 0
            [gradient,Loss_MSE,Loss_Physics] = dlfeval(accfun,...
                parameters,PICAE,dlP_0_now,dlP_1_now,...
                alpha,Model);
    
        elseif PICAE.SelfModelling == 1 && PICAE.UseObservale == 0
            [gradient,Loss_MSE,Loss_Physics] = dlfeval(accfun,...
                parameters,PICAE,dlP_0_now,dlP_1_now,...
                alpha,Model);
    
        else
            [gradient,Loss_MSE,Loss_Physics] = dlfeval(accfun,...
                parameters,PICAE,dlP_0_now,dlP_1_now,...
                alpha,Model,...
                dlPc,dlOc(1,:),dlOc(2,:));
           
        end

        % Learning rate
        LearningRate = PICAE.LearningRate0./(1+PICAE.DecayRate0*iteration);

        %ADAM
        [parameters,averageGrad,averageSqGrad] = adamupdate(parameters,gradient,averageGrad, ...
            averageSqGrad,iteration,LearningRate);

    end

    %% Plotting
    
    if Config.plot == 1

        if epoch/Config.plot_rate == floor(epoch/Config.plot_rate)

            % plot losses
            figure(1)
            subplot(2,2,1)
            plot(epoch,(Loss_MSE + alpha*Loss_Physics)./(1+alpha),'.k','MarkerSize',12)
            plot(epoch,Loss_MSE,'.b','MarkerSize',12)
            plot(epoch,(alpha*Loss_Physics)./(1+alpha),'.r','MarkerSize',12)

            % prediction
            [dlP_0_p,Code0] = AE_network_LV(dlP_0(:,Config.plot_ind),[],1,parameters,PICAE,0);
                
            subplot(2,2,2)
            hold off
            plot(dlP_0(1,Config.plot_ind))
            hold on
            plot(dlP_0_p(1,Config.plot_ind))
            grid on
            grid minor
            xlabel("time")
            ylabel("Reconstructed Query p")
            legend("input","predicted")

            subplot(2,2,4)
            hold off
            plot(dlP_0(1+PICAE.Buffer_Size,Config.plot_ind))
            hold on
            plot(dlP_0_p(1+PICAE.Buffer_Size,Config.plot_ind))
            grid on
            grid minor
            xlabel("time")
            ylabel("Reconstructed Query q")
            legend("input","predicted")

            subplot(2,2,3)
            hold off
            plot(LV.x(Config.plot_ind),LV.y(Config.plot_ind))
            hold on
            plot(Code0(1,:),Code0(2,:))
            if PICAE.UseObservale == 1 && PICAE.SelfModelling == 1
                plot(dlOc(1,:),dlOc(2,:),'xk','LineWidth',1.2,'MarkerSize',10)
            end
            grid on
            grid minor
            xlabel("x")
            ylabel("y")
            legend("real","reconstructed")
            
            drawnow

        end
    end

    if PICAE.SelfModelling == 1
        
        coeff = [double(extractdata(gather(parameters.model.a)));...
            double(extractdata(gather(parameters.model.b)));...
            double(extractdata(gather(parameters.model.c)));...
            double(extractdata(gather(parameters.model.d)))];

        disp("coefficients: ")
        disp(floor(coeff'*100)/100)

    end

    %% New alpha (if adaptive method)

    if PICAE.alpha_adaptive == 1
        
        MSE = double(extractdata(gather(Loss_MSE)));
        alpha = alpha.*(1 - 0.1*tanh(MSE./PICAE.MSE_target-1));

        alpha = max(alpha,PICAE.alpha_range(1));
        alpha = min(alpha,PICAE.alpha_range(2));

        disp("alpha :")
        disp(alpha)

    end

    

end





