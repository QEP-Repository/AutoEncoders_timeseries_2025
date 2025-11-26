function Data = CausAE_TrainingTestSet(Data,CausAE)

%% Prepare indices

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


%% Prepare Training and Test data

Data.train_input_all = Data.Input_all(:,ind_train);
Data.train_input_wod = Data.Input_wod(:,ind_train);
Data.train_output = Data.Output(:,ind_train);

Data.test_input_all = Data.Input_all(:,ind_test);
Data.test_input_wod = Data.Input_wod(:,ind_test);
Data.test_output = Data.Output(:,ind_test);

end

