function Data = CausAE_Buffer(Data,CausAE)

% indices condidering time delays

N = length(Data.(CausAE.input_all(1)));
max_delay = max(CausAE.time_step);

ind_output = 1 : N;

for i = 1 : length(CausAE.input_all)
    ind_delay(i,:) = ind_output-CausAE.time_step(i)-CausAE.time_window(i);
end

ind_to_remove = find(min(ind_delay,[],1)<=0);

ind_delay(:,ind_to_remove) = [];

ind_delay = ind_delay + CausAE.time_window';

ind_output(:,ind_to_remove) = [];

% All inputs

Input_all = [];

for i = 1 : length(CausAE.input_all)

    xx = Data.(CausAE.input_all(i));

    yy = buffer(xx,CausAE.time_window(i),CausAE.time_window(i)-CausAE.buffer_step(i));

    Input_all = [Input_all; ...
        yy(:,ind_delay(i,:))];

end

% Inputs w/o driver

Input_wod = [];

for i = 1 : length(CausAE.input_wod)

    j = find(CausAE.input_all == CausAE.input_wod(i));

    xx = Data.(CausAE.input_wod(i));

    yy = buffer(xx,CausAE.time_window(j),CausAE.time_window(j)-CausAE.buffer_step(j));

    Input_wod = [Input_wod; yy(:,ind_delay(j,:))];

end

% Outputs

Output = [];

for i = 1 : length(CausAE.effect)

    j = find(CausAE.input_all == CausAE.effect(i));

    xx = Data.(CausAE.effect(i));

    yy = buffer(xx,CausAE.time_window(j),CausAE.time_window(j)-CausAE.buffer_step(j));

    Output = [Output; yy(:,ind_output)];

end

Data.Input_all = Input_all;
Data.Input_wod = Input_wod;
Data.Output = Output;


end




