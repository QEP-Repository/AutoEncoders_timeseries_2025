
gpu = gpuDevice();
reset(gpu)

gpu = gpuDevice();
disp(gpu)
wait(gpu)

clear all; clc;

%%

addpath Functions\

load 'Numerical Data'\LotkaVolterra.mat

t = LV.t;
% 
% % Case A
% x = LV.x;
% y = LV.y;

% Case B

xb = LV.x;
yb = LV.y;

x = (LV.x+LV.y)/2;
y = LV.x./(1+LV.y);

MSE_target = (0.10*mean(x) + 0.10*mean(y))*1.5;

x = normrnd(x,0.10*mean(x));
y = normrnd(y,0.10*mean(y));

Model.a = dlarray(LV.a,'CB');
Model.b = dlarray(LV.b,'CB');
Model.c = dlarray(LV.c,'CB');
Model.d = dlarray(LV.d,'CB');
Model.dt = dlarray(LV.dt,'CB');

%% Config

AE.Buffer_Size = 20;
AE.Buffer_Res = 1;
AE.MiniBatch = 2000;
AE.PhysicsBatch = 2000;
AE.step = 1; 

AE.LearningRate0 = 1e-3;
AE.DecayRate0 = 0;

%% Prepare Buffer

X_all = [buffer(x,AE.Buffer_Size,AE.Buffer_Size-AE.Buffer_Res);...;
        buffer(y,AE.Buffer_Size,AE.Buffer_Size-AE.Buffer_Res)];

B_all = [buffer(xb,AE.Buffer_Size,AE.Buffer_Size-AE.Buffer_Res);...;
        buffer(yb,AE.Buffer_Size,AE.Buffer_Size-AE.Buffer_Res)];

xb_all = B_all(11,:); yb_all = B_all(31,:);

clear B_all

inds = AE.Buffer_Size:length(x);

X = X_all(:,inds);

inds_b = inds([200 300 400 500 1000 1100 1200 1250 1700 1800]);

Xb = X_all(:,inds_b);
xb = xb_all(inds_b); clear xb_all
yb = yb_all(inds_b); clear yb_all

%% Prepare 

% no delay
inds_0 = inds(1)+AE.step:inds(end);
X_0 = X_all(:,inds_0);

% 1 delay
inds_1 = inds(1):inds(end)-AE.step;
X_1 = X_all(:,inds_1);

Model.dt = Model.dt.*AE.step;

%% Deep Learning Variables

dlX_1 = dlarray(X_1,'CB');
dlX_0 = dlarray(X_0,'CB');

dlx_b = dlarray(xb,'CB');
dly_b = dlarray(yb,'CB');
dlX_0b = dlarray(Xb,'CB');

%% AutoEncoder Initialisation

AE.Layer_En = [20 20 20 20 20 20 20];
AE.Layer_Dec = flip(AE.Layer_En);
AE.CodeSize = 2;

[~,~,parameters] = AE_network_LV_SM(dlX_0,dlX_0,0,[],AE,0);
[dlY0,Code] = AE_network_LV_SM(dlX_0,[],1,parameters,AE,0);

averageGrad = [];
averageSqGrad = [];

%% Model gradient acceleration

accfun = dlaccelerate(@ModelGradient_LV_SM_constrainxy);

%% Options

alpha = dlarray(1e5);
alpha_min = 1;
alpha_max = 1e10;

%% Training

figure(1)
clf

iteration = 0;

for epoch = 1 : 1e4

    for i = 1 : 100

        % iteration update
        iteration = iteration + 1;

        % mini batch extraction
        ind_batch = randsample(size(dlX_0,2),AE.MiniBatch);
        % ind_batch = 1 : 3e3;

        dlX0_now = dlX_0(:,ind_batch);
        dlX1_now = dlX_1(:,ind_batch);

        % model gradient
        [gradients, MSE, Physics,MSExy] = dlfeval(accfun,...
            parameters,AE,dlX0_now,dlX1_now,0.1,alpha,Model,dlX_0b,dlx_b,dly_b);

        % Learning rate
        LearningRate = AE.LearningRate0./(1+AE.DecayRate0*iteration);

        %ADAM
        [parameters,averageGrad,averageSqGrad] = adamupdate(parameters,gradients,averageGrad, ...
            averageSqGrad,iteration,LearningRate);

    end

    %%

    disp("epoch = " + epoch)

    Loss = (MSE + alpha.*Physics + alpha.*MSExy)./(1+alpha);

    ind_plot = 1:4000;
    dlX0_plot = dlX_0(:,ind_plot);
    [X0_p,Code] = AE_network_LV_SM(dlX0_plot,[],1,parameters,AE,0);

    figure(1)
    subplot(2,3,1)
    plot(epoch,Loss,'.b','markersize',16)
    hold on
    set(gca,'YScale','log')

    subplot(2,3,2)
    plot(epoch,MSE,'.b','markersize',16)
    hold on
    plot(epoch,alpha.*Physics,'.r','markersize',16)
    plot(epoch,alpha.*MSExy,'.k','markersize',16)
    set(gca,'YScale','log')

    subplot(2,3,3)
    plot(Code(1,:),Code(2,:),'-b')

    subplot(2,3,4)
    hold off
    plot(LV.t(ind_plot),LV.x(ind_plot),'-b')
    hold on
    plot(LV.t(ind_plot),LV.y(ind_plot),'-r')
    plot(LV.t(ind_plot),Code(1,:),'-.b')
    plot(LV.t(ind_plot),Code(2,:),'-.r')
    plot(LV.t(inds_b),LV.x(1,inds_b),'.k','MarkerSize',12)
    plot(LV.t(inds_b),LV.y(1,inds_b),'.k','MarkerSize',12)
    xlim([-inf inf])

    subplot(2,3,5)
    hold off
    plot(t(ind_plot),dlX0_plot(1,:),'-b')
    hold on
    plot(t(ind_plot),dlX0_plot(AE.Buffer_Size+1,:),'-r')
    plot(t(ind_plot),X0_p(1,:),'-.b')
    plot(t(ind_plot),X0_p(AE.Buffer_Size+1,:),'-.r')
    xlim([-inf inf])

    subplot(2,3,6)
    hold off
    plot(dlX0_plot(1,:),X0_p(1,:),'.b')
    hold on
    plot(dlX0_plot(AE.Buffer_Size+1,:),X0_p(AE.Buffer_Size+1,:),'.r')
    plot([0 max(dlX0_plot(1,:))],[0 max(dlX0_plot(1,:))],'-.k')
    xlim([-inf inf])

    drawnow

    %%
    if (MSE + MSExy) < MSE_target
        alpha = alpha.*1.1;
        alpha = min(alpha,alpha_max);
    else
        alpha = alpha.*0.9;
        alpha = max(alpha,alpha_min);
    end



    %%

    Coefficients = [parameters.model.a; parameters.model.b; ...
        parameters.model.c; parameters.model.d; alpha];

    disp(Coefficients)

end





