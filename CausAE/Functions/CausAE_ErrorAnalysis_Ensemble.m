
function Results = CausAE_ErrorAnalysis_Ensemble(Data,CausAE,Ensemble)

%% Take Data

Input_All = dlarray(Data.Input_all,'CB');
Input_wod = dlarray(Data.Input_wod,'CB');
Output_All = Data.Output;

ind_all = 1:length(Data.t);

%%

N = size(Data.t,2); M = size(Data.Output,2);

ind_train = [];

for i = 1 : size(CausAE.training_window,1)

    ind_train = [ind_train ...
        floor((N.*CausAE.training_window(i,1)+1):(N.*CausAE.training_window(i,2)))];

end

ind_test = 1 : N;
ind_test(ind_train) = [];

ind_train = ind_train + M - N; ind_train(ind_train<1) = []; 
ind_test = ind_test + M - N; ind_test(ind_test<1) = [];

%% Target Reconstructed

% Index for Reverse Buffer
j = find(CausAE.input_all == CausAE.effect);

y_output = ReverseBuffer(Output_All,CausAE.time_window(j),CausAE.time_window(j)-CausAE.buffer_step(j));

%% Correct inds

ind_all(1:length(Data.t)-length(y_output)) = [];

%% Prediction with Ensemble

for e1 = 1 : CausAE.Ensembles

    %% All Predictors
    
    % Predict with all 
    y_all = CausAE_Network_1(Input_All,[],1,Ensemble{e1}.Model_All.parameters,CausAE,0);
    y_all = double(extractdata(gather(y_all)));

    % Reverse buffer
    y_all = ReverseBuffer(y_all,CausAE.time_window(j),CausAE.time_window(j)-CausAE.buffer_step(j));

    % Put all prediction from ensemble togheter
    y_all_en(e1,:) = y_all; 

    %% WOD Predictors
    
    % Predict with all 
    y_wod = CausAE_Network_1(Input_wod,[],1,Ensemble{e1}.Model_wod.parameters,CausAE,0);
    y_wod = double(extractdata(gather(y_wod)));

    % Reverse buffer
    y_wod = ReverseBuffer(y_wod,CausAE.time_window(j),CausAE.time_window(j)-CausAE.buffer_step(j));

    % Put all prediction from ensemble togheter
    y_wod_en(e1,:) = y_wod; 

end

%% Evaluate Statistics

y_all_mu = median(y_all_en,1);
y_all_std = std(y_all_en,[],1);

y_wod_mu = median(y_wod_en,1);
y_wod_std = std(y_wod_en,[],1);

%% Stastics single value of Ensemble

E2_all = (y_all_en(:,ind_test)-y_output(ind_test)).^2;
E2_wod = (y_wod_en(:,ind_test)-y_output(ind_test)).^2;

Results.Ensemble.MSE_all_test = mean(E2_all,2);
Results.Ensemble.MSE_wod_test = mean(E2_wod,2);

Results.Ensemble.MSElog_all_test = mean(log10(E2_all),2);
Results.Ensemble.MSElog_wod_test = mean(log10(E2_wod),2);

%% Statistics 

Results.Data.ind_all = ind_all;
Results.Data.ind_train = ind_train;
Results.Data.ind_test = ind_test;

Results.Data.y_target = y_output;
Results.Data.y_all_mu = y_all_mu;
Results.Data.y_all_std = y_all_std;
Results.Data.y_wod_mu = y_wod_mu;
Results.Data.y_wod_std = y_wod_std;

E2_max = (y_output-mean(y_output)).^2;

E2_all = (y_all_mu-y_output).^2;
E2_wod = (y_wod_mu-y_output).^2;

Z2_all = (y_all_mu-y_output).^2./(y_all_std).^2;
Z2_wod = (y_wod_mu-y_output).^2./(y_wod_std).^2;

E2_all_sort = sort(E2_all);
E2_wod_sort = sort(E2_wod);

E2_delta = mean(E2_wod_sort-E2_all_sort);
E2_prevalence = mean((E2_wod_sort-E2_all_sort)>0);

Results.Statistics.MSE_all = mean(E2_all);      
Results.Statistics.MSE_wod = mean(E2_wod);

Results.Statistics.Z_all = median(Z2_all);
Results.Statistics.Z_wod = median(Z2_wod);

Results.Statistics.R2_all = 1 - mean(E2_all)./mean((y_output-mean(y_output)).^2);
Results.Statistics.R2_wod = 1 - mean(E2_wod)./mean((y_output-mean(y_output)).^2);

Results.Statistics.E2_delta = E2_delta;
Results.Statistics.E2_prevalence = E2_prevalence;

%% Causality Decision 

% you can implement other statistical tests, here we the difference of mean
% of error

Results.Tests.Z = (mean(Results.Ensemble.MSE_wod_test)-mean(Results.Ensemble.MSE_all_test))./...
    (std(Results.Ensemble.MSE_wod_test)+std(Results.Ensemble.MSE_all_test));
Results.Tests.Decision = Results.Tests.Z>2;

%% Disp results

disp(Results.Tests)

end




