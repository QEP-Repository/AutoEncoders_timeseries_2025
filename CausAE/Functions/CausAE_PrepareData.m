
function Data = CausAE_PrepareData(Data,CausAE)

N = Data.N;

%% From time series to time windows

% X time windows
X = buffer(Data.x,CausAE.time_window(1),CausAE.time_window(1)-CausAE.buffer_step(1));

% Y time windows
Y = buffer(Data.y,CausAE.time_window(2),CausAE.time_window(2)-CausAE.buffer_step(2));

%% Prepare dlarray for training and test

ind_train_window = floor(CausAE.training_window.*Data.N);

ind_train_input_x = [];
ind_train_input_y = [];
ind_train_output = [];

delta_max = max(CausAE.time_step);
delta = CausAE.time_step;

for i = 1 : size(ind_train_window,1)

    ind_train_temp_x_in = ind_train_window(i,1)-delta(1):ind_train_window(i,2)-delta(1);
    ind_train_temp_y_in = ind_train_window(i,1)-delta(2):ind_train_window(i,2)-delta(2);
    ind_train_temp_out = ind_train_window(i,1):ind_train_window(i,2);

    ind_to_remove = find((ind_train_temp_x_in<=0)|(ind_train_temp_y_in<=0)|(ind_train_temp_out<=0));

    ind_train_temp_x_in(ind_to_remove) = [];
    ind_train_temp_y_in(ind_to_remove) = [];
    ind_train_temp_out(ind_to_remove) = [];

    ind_train_input_x = [ind_train_input_x ind_train_temp_x_in];
    ind_train_input_y = [ind_train_input_y ind_train_temp_y_in];
    ind_train_output = [ind_train_output ind_train_temp_out];

end

clear ind_train_temp_out ind_train_temp_x_in ind_train_temp_y_in

% training data

tx_train_in = Data.t(ind_train_input_x);
ty_train_in = Data.t(ind_train_input_y);
t_train_out = Data.t(ind_train_output);

X_train_in = X(:,ind_train_input_x);
Y_train_in = Y(:,ind_train_input_y);
X_train_out = X(:,ind_train_output);
Y_train_out = Y(:,ind_train_output);

% test data

ind_test_input_x = 0-delta(1):N-delta(1);
ind_test_input_y = 0-delta(2):N-delta(2);

ind_test_output = 0:N;
ind_test_output(ind_train_output)=[];
ind_test_input_x = ind_test_output-delta(1);
ind_test_input_y = ind_test_output-delta(2);

ind_to_remove = find((ind_test_input_x<=0)|(ind_test_input_y<=0)|(ind_test_output<=0));

ind_test_input_x(ind_to_remove) = [];
ind_test_input_y(ind_to_remove) = [];
ind_test_output(ind_to_remove) = [];

X_test_in = X(:,ind_test_input_x);
Y_test_in = Y(:,ind_test_input_y);
X_test_out = X(:,ind_test_output);
Y_test_out = Y(:,ind_test_output);

% dlarray

dX_train_in = dlarray(X_train_in,'CB');
dY_train_in = dlarray(Y_train_in,'CB');
dX_train_out = dlarray(X_train_out,'CB');
dY_train_out = dlarray(Y_train_out,'CB');

dX_test_in = dlarray(X_test_in,'CB');
dY_test_in = dlarray(Y_test_in,'CB');
dX_test_out = dlarray(X_test_out,'CB');
dY_test_out = dlarray(Y_test_out,'CB');

Data.train_dX_in = dX_train_in;
Data.train_dY_in = dY_train_in;
Data.train_dX_out = dX_train_out;
Data.train_dY_out = dY_train_out;

Data.test_dX_in = dX_test_in;
Data.test_dY_in = dY_test_in;
Data.test_dX_out = dX_test_out;
Data.test_dY_out = dY_test_out;

end
