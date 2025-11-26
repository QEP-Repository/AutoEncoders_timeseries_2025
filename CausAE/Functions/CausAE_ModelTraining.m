function Model = CausAE_ModelTraining(Data,CausAE,dIn_train,dOut_train,dIn_test,Out_test)

ind = randsample(size(dIn_train,2),1000);

[~,~,parameters] = CausAE_Network_1(dIn_train(:,ind),dOut_train(:,ind),0,[],CausAE,0);
averageGrad = [];
averageSqGrad = [];

% [dOutp,Code] = CausAE_Network_1(dIn_train(:,ind),[],1,parameters,CausAE,0);

%% Model Gradient

accfun = dlaccelerate(@CausAE_ModelGradient);

%% Training

figure(1); clf;

iteration = 0;
N = size(dIn_train,2);

Validation_checks = 0;
Loss_best = mean((Out_test).^2,'all');

for epoch = 1 : CausAE.max_epoch

    for i = 1 : CausAE.iter_per_epoch

        % iteration update
        iteration = iteration + 1;

        % mini batch extraction
        ind_batch = randsample(N,CausAE.minibatch);
        dIn_batch = dIn_train(:,ind_batch);
        dOut_batch = dOut_train(:,ind_batch);

        % model gradient
        [gradients, Loss_MSE, Loss_Code] = dlfeval(accfun,dIn_batch,...
            dOut_batch,parameters,CausAE,CausAE.alpha,CausAE.sigma);

        % Learning rate
        LearningRate = CausAE.LearningRate0./(1+CausAE.DecayRate0*iteration);

        %ADAM
        [parameters,averageGrad,averageSqGrad] = adamupdate(parameters,gradients,averageGrad, ...
            averageSqGrad,iteration,LearningRate);

    end

    %% Plot Results epoch 

    [dYp,Code] = CausAE_Network_1(dIn_test,[],1,parameters,CausAE,0);

    Yp = double(extractdata(gather(dYp)));
    
    MSE = mean((Yp-Out_test).^2,'all');

    % MSE = mean((yp_test-y_test_restored).^2);

    figure(1)
    subplot(2,2,1)
    plot(epoch,Loss_MSE,'.b','MarkerSize',16)
    hold on
    plot(epoch,MSE,'.r','MarkerSize',16)
    set(gca,'YScale','log')
    % 
    % subplot(2,2,4)
    % plot(epoch,MSE,'.b','MarkerSize',16)
    % hold on
    % set(gca,'YScale','log')
   
    subplot(2,2,3)
    plot(Yp(:),Out_test(:),'.r','linewidth',1.2)

    if size(Code,1) > 1
        subplot(2,2,2)
        plot(Code(1,:),Code(2,:),'.b','markersize',12)
        grid on
        grid minor
        xlabel("Code 1")
        ylabel("Code 2")
    end

    drawnow

    if MSE < Loss_best

        Model.parameters = parameters;
        Model.epoch = epoch;

        Validation_checks = 0;

        Loss_best = MSE;

    else

        Validation_checks = Validation_checks + 1;
        
        if Validation_checks > CausAE.MaxValidationChecks 
            disp("Validation Check # " + Validation_checks)
            break
        end

    end

end



end