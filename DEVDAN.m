% LicenseCC BY-NC-SA 4.0
% 
% Copyright (c) 2018 Andri Ashfahani Mahardhika Pratama



















%% list of equation
% equation 8 is implemented in the line 286
% equation 10 is implemented in the line  294
% equation 11 is implemented in the line 298
% equation 12, 14 is implemented in the line 299
% equation 5 is implemented in the lines 306 and 365
% equation 15 is implemented in the line 329
% equation 16 is implemented in the line 387
% equation 19 is implemented in the line 394
% equation 20 is implemented in the line 395
% equation 1 is implemented in the lines 420 and 421
% equation 2 is implemented in the lines 422 and 425

%% main code
function [parameter,performance] = DEVDAN(data,I,portionLabeledData,...
    mode,selectiveSample,chunkSize)
%% divide the data into nFolds chunks
fprintf('=========DEVDAN is started=========\n')
[nData,mn] = size(data);
M = mn - I;
l = 0;
nFolds = round(size(data,1)/chunkSize);                 % number of data chunk
chunk_size = round(nData/nFolds);
round_nFolds = floor(nData/chunk_size);
Data = {};
if round_nFolds == nFolds
    if nFolds == 1
        Data{1} = data;
    else
        for i=1:nFolds
            l=l+1;
            Data1 = data(((i-1)*chunk_size+1):i*chunk_size,:);
            Data{l} = Data1;
        end
    end
else
    if nFolds == 1
        Data{1} = data;
    else
        for i=1:nFolds-1
            l=l+1;
            Data1 = data(((i-1)*chunk_size+1):i*chunk_size,:);
            Data{l} = Data1;
        end
        foldplus = randperm(nFolds-1,1);
        Data{nFolds} = Data{foldplus};
    end
end
tTest = [];
clear data Data1

%% initiate model
K = 1;
if mode == 2
    K = M;
end
parameter.net       = netInit([I K M]);
parameter.net.index = 1;
parameter.net.mode  = mode;

%% initiate DAE parameter
parameter.dae{1}.lr     = 0.001;
parameter.dae{1}.K      = K;
parameter.dae{1}.Kg     = 1;
parameter.dae{1}.kk     = 0;
parameter.dae{1}.node   = [];
parameter.dae{1}.BIAS2  = [];
parameter.dae{1}.VAR    = [];
parameter.dae{1}.Loss   = [];
parameter.dae{1}.miu_A_old  = 0;
parameter.dae{1}.var_A_old  = 0;
parameter.dae{1}.miu_NS_old = 0;
parameter.dae{1}.var_NS_old = 0;
parameter.dae{1}.miu_NHS_old = 0;
parameter.dae{1}.var_NHS_old = 0;
parameter.dae{1}.miumin_NS   = [];
parameter.dae{1}.miumin_NHS  = [];
parameter.dae{1}.stdmin_NS   = [];
parameter.dae{1}.stdmin_NHS  = [];
parameter.mode = 'sigmsigm';

%% initiate node evolving iterative parameters
parameter.ev{1}.kp = 0;
parameter.ev{1}.K  = K;
parameter.ev{1}.Kd = 0;
parameter.ev{1}.node  = [];
parameter.ev{1}.noded = [];
parameter.ev{1}.BIAS2 = [];
parameter.ev{1}.VAR   = [];
parameter.ev{1}.miu_x_old  = 0;
parameter.ev{1}.var_x_old  = 0;
parameter.ev{1}.miu_NS_old = 0;
parameter.ev{1}.var_NS_old = 0;
parameter.ev{1}.miu_NHS_old = 0;
parameter.ev{1}.var_NHS_old = 0;
parameter.ev{1}.miumin_NS   = [];
parameter.ev{1}.miumin_NHS  = [];
parameter.ev{1}.stdmin_NS   = [];
parameter.ev{1}.stdmin_NHS  = [];

%% main loop, prequential evaluation
for iFolds = 1:nFolds
    %% load the data chunk-by-chunk
    [bd,~] = size(Data{iFolds}(:,I+1:mn));
    x = Data{iFolds}(:,1:I);
    T = Data{iFolds}(:,I+1:mn);
    clear Data{t}
    
    %% neural network testing
    start_test = tic;
    fprintf('=========Chunk %d of %d=========\n', iFolds, size(Data,2))
    disp('Discriminative Testing: running ...');
    parameter.net.t = iFolds;
    parameter.net   = testing(parameter.net,x,T,parameter.ev);
    
    %% metrics calculation
    parameter.ev{parameter.net.index}.t = iFolds;
    parameter.Loss(iFolds)              = parameter.net.loss(parameter.net.index);
    tTest(bd*iFolds+(1-bd):bd*iFolds,:) = parameter.net.activityOutput{1};
    act(bd*iFolds+(1-bd):bd*iFolds,:)   = parameter.net.actualLabel;
    out(bd*iFolds+(1-bd):bd*iFolds,:)   = parameter.net.classPerdiction;
    parameter.cr(iFolds)                = parameter.net.classRate;
    ClassificationRate(iFolds)          = mean(parameter.cr);
    parameter.residual_error(bd*iFolds+(1-bd):bd*iFolds,:) = parameter.net.residual_error;
    fprintf('Classification rate %d\n', ClassificationRate(iFolds))
    disp('Discriminative Testing: ... finished');
    
    %% statistical measure
    [parameter.net.f_measure(iFolds,:),parameter.net.g_mean,parameter.net.recall(iFolds,:),parameter.net.precision(iFolds,:),parameter.net.err(iFolds,:)] = performanceMeasure(parameter.net.actualLabel, parameter.net.classPerdiction, M);
    if iFolds == nFolds - 1
        fprintf('=========DEVDAN is finished=========\n')
        break               % last chunk only testing
    end
    parameter.net.test_time(iFolds) = toc(start_test);
    
    %% Generative training
    start_train = tic;
    if mode ~= 1
        [parameter,~] = generativeTraining(x,parameter);
        parameter.dae{1}.Loss(iFolds) = parameter.dae{1}.LF;
        
        %% calculate bias^2/var based on generative training
        parameter.bs2g(iFolds) = sum(parameter.dae{parameter.net.index}.BIAS2(bd*iFolds+(1-bd):bd*iFolds,:))/bd;
        parameter.varg(iFolds) = sum(parameter.dae{parameter.net.index}.VAR  (bd*iFolds+(1-bd):bd*iFolds,:))/bd;
        
        %% calculate hidden node evolution based on generative training
        parameter.ndcg(iFolds) = parameter.dae{parameter.net.index}.K;
    end
    
    %% Discrinimanive training
    disp('Discriminative Training: running ...');
    parameter = discriminativeTraining(parameter,T,portionLabeledData,...
        selectiveSample);
    disp('Discriminative Training: ... finished');
    parameter.net.update_time(iFolds) = toc(start_train);
    
    %% calculate hidden node evolution based on discriminative training
    parameter.ndcd(iFolds) = parameter.ev{parameter.net.index}.K;
    
    %% clear current chunk data
    clear Data{t}
    parameter.net.activity = {};
end
clc

%% statistical measure
[performance.f_measure,performance.g_mean,performance.recall,performance.precision,performance.err] = performanceMeasure(act, out, M);

parameter.nFolds = nFolds;
performance.update_time         = [sum(parameter.net.update_time) mean(parameter.net.update_time) std(parameter.net.update_time)];
performance.test_time           = [sum(parameter.net.test_time) mean(parameter.net.test_time) std(parameter.net.test_time)];
performance.classification_rate = [mean(parameter.cr(2:end)) std(parameter.cr(2:end))]*100;
performance.LayerWeight         = parameter.net.beta;
performance.NoOfnode            = [mean(parameter.ev{1}.node) std(parameter.ev{1}.node)];
performance.NumberOfParameters  = parameter.net.mnop;
performance.compressionRate     = parameter.ev{1}.kp/parameter.dae{1}.kk;
fprintf('=========DEVDAN is finished=========\n')
end

%% evolving DAE
function [parameter,h] = generativeTraining(x,parameter)
ly = parameter.net.index;
% [N,~] = size(x);

% %% add adversarial sample
% if parameter.net.t > 2
%     sign_grad_Loss = sign(parameter.dae{ly}.Loss(parameter.net.t-1) - parameter.dae{ly}.Loss(parameter.net.t-2));
%     adversarial_samples = x + 0.007*sign_grad_Loss;
%     kk = randperm(N);
%     adversarial_samples = adversarial_samples(kk,:);
%     kk = randperm(N);
%     x = x(kk,:);
%     x = [x;adversarial_samples];
% end

%% initiate parameter
[nData,~] = size(x);
s       = RandStream('mt19937ar','Seed',0);
kk      = randperm(s,nData);
x       = x(kk,:);
[M,~]   = size(parameter.net.weightSoftmax{1});
[~,I]   = size(x);
W       = parameter.net.weight{ly}(:,2:end);
b       = parameter.net.weight{ly}(:,1);
c       = parameter.net.c{ly};
[K,~]   = size(W);
mode    = parameter.mode;

%% a parameter to indicate if there is growing/pruning
grow  = 0;
prune = 0;

%% initiate performance matrix
miu_A_old   = parameter.dae{ly}.miu_A_old;
var_A_old   = parameter.dae{ly}.var_A_old;
miu_NS_old  = parameter.dae{ly}.miu_NS_old;
var_NS_old  = parameter.dae{ly}.var_NS_old;
miu_NHS_old = parameter.dae{ly}.miu_NHS_old;
var_NHS_old = parameter.dae{ly}.var_NHS_old;
miumin_NS   = parameter.dae{ly}.miumin_NS;
miumin_NHS  = parameter.dae{ly}.miumin_NHS;
stdmin_NS   = parameter.dae{ly}.stdmin_NS;
stdmin_NHS  = parameter.dae{ly}.stdmin_NHS;
nodeg       = parameter.dae{ly}.node;
Kg = parameter.dae{ly}.Kg;
lr = parameter.dae{ly}.lr;
kk = parameter.dae{ly}.kk;
node  = parameter.ev{ly}.node;
BIAS2 = parameter.dae{ly}.BIAS2;
VAR   = parameter.dae{ly}.VAR;

%% main loop devdann
x_tail  = x;
kprune  = 0;
kgrow   = 0;

%% Generative training
disp('DEVDAN Training: running ...');
for iData = 1:nData
    kk = kk + 1;
    
    %% Input masking
    maskingLevel    = 0.1;
    x_tail(iData,:) = maskingnoise(x_tail(iData,:),I,maskingLevel);
    
    %% feedforward #1
    a = W*x_tail(iData,:)' + b;
    h = sigmf(a,[1,0]);
    a_hat = W'*h + c;
    switch mode
        case 'sigmsigm'
            x_hat = sigmf(a_hat,[1,0]);
        case 'sigmafn'
            x_hat = a_hat;
    end
    x_hat = x_hat';
    
    %% calculate error
    e(iData,:) = x(iData,:) - x_hat;
    L(iData,:) = 0.5*norm(e(iData,:))^2;
    
    if parameter.net.index == parameter.net.nHiddenLayer
        %% Expectation of z
        A = a;         % calculate A, equation 8
        
        %% Incremental calculation of x_tail mean and variance
        [miu_A,std_A,var_A] = meanstditer(miu_A_old,var_A_old,A,kk);
        miu_A_old = miu_A;
        var_A_old = var_A;
        
        py = probit(miu_A,std_A);
        Ey = sigmf(py,[1,0]);                        % equation 10
        
        switch mode
            case 'sigmsigm'
                Ez  = sigmf(W'*Ey + c,[1,0]);        % equation 11
                Ez2 = sigmf(W'*Ey.^2 + c,[1,0]);     % equation 12, 14
            case 'sigmafn'
                Ez  = W'*Ey + c;
                Ez2 = W'*Ey.^2 + c;
        end
        
        %% Network mean calculation
        bias2   = (Ez - x(iData,:)').^2;          % equation 5, Bias2
        ns      = bias2;
        NS      = mean(ns);                       % use mean to summarize
        
        %% Incremental calculation of mean bias and std bias
        [miu_NS,std_NS,var_NS] = meanstditer(miu_NS_old,var_NS_old,NS,kk);
        BIAS2(kk,:) = miu_NS;
        miu_NS_old  = miu_NS;   % mean bias
        var_NS_old  = var_NS;   % std2 bias
        miustd_NS   = miu_NS + std_NS;
        if kk <= 1 || grow == 1
            miumin_NS = miu_NS;  % reset miu min of bias
            stdmin_NS = std_NS;  % reset std min of bias
        else
            if miu_NS < miumin_NS
                miumin_NS = miu_NS;     % update miu min of bias
            end
            if std_NS < stdmin_NS
                stdmin_NS = std_NS;     % update std min of bias
            end
        end
        switch mode
            case 'sigmsigm'
                miustdmin_NS = miumin_NS + (1.3*exp(-NS)+0.7)*stdmin_NS;    % right hand side term of equation 15
            case 'sigmafn'
                miustdmin_NS = miumin_NS + (1.3*exp(-NS)+0.7)*stdmin_NS;
        end
        
        %% growing hidden unit if 15 is satisfied
        if miustd_NS >= miustdmin_NS && kk > 1 && parameter.net.mode ~= 2
            grow = 1;
            K    = K + 1;
            Kg   = Kg + 1;
            fprintf('The new node no %d is FORMED around sample %d\n', K, iData)
            kgrow       = kgrow + 1;
            node(kk)    = K;
            nodeg(kk)   = Kg;
            b_init      = rand(1);
            if b_init < 0.5
                binit = -1;
            elseif b_init > 0.5
                binit = 1;
            end
            b = [b;binit];              % new bias randomized: -1 or 1
            W = [W;-e(iData,:)];        % new weight from -error
            miu_A_old = [miu_A_old;0];
            var_A_old = [var_A_old;0];
            parameter.net.velocity{ly}          = [parameter.net.velocity{ly};zeros(1,I+1)];
            parameter.net.grad{ly}              = [parameter.net.grad{ly};zeros(1,I+1)];
            parameter.net.weightSoftmax{ly}     = [parameter.net.weightSoftmax{ly} normrnd(0,sqrt(2/(K+1)),[parameter.net.initialConfig(end),1])];
            parameter.net.momentumSoftmax{ly}   = [parameter.net.momentumSoftmax{ly} zeros(M,1)];
            parameter.net.gradSoftmax{ly}       = [parameter.net.gradSoftmax{ly} zeros(M,1)];
        else
            grow = 0;
            node(kk) = K;
            nodeg(kk) = Kg;
        end
        
        %% Network variance calculation
        var = Ez2 - Ez.^2;          % equation 5, Bias2
        NHS = mean(var);            % use mean to summarize
        
        %% Incremental calculation of mean variance and std variance
        [miu_NHS,std_NHS,var_NHS] = meanstditer(miu_NHS_old,var_NHS_old,NHS,kk);
        VAR(kk,:)   = miu_NHS;
        miu_NHS_old = miu_NHS;
        var_NHS_old = var_NHS;
        miustd_NHS  = miu_NHS + std_NHS;
        if kk <= I + 1 || prune == 1
            miumin_NHS = miu_NHS;       % reset miu min of variance
            stdmin_NHS = std_NHS;       % reset std min of variance
        else
            if miu_NHS < miumin_NHS
                miumin_NHS = miu_NHS;   % update miu min of variance
            end
            if std_NHS < stdmin_NHS
                stdmin_NHS = std_NHS;   % update std min of variance
            end
        end
        switch mode
            case 'sigmsigm'
                miustdmin_NHS = miumin_NHS + 2*(1.3*exp(-NHS)+0.7)*stdmin_NHS;      % right hand side term of equation 16
            case 'sigmafn'
                miustdmin_NHS = miumin_NHS + 2*(1.3*exp(-NHS)+0.7)*stdmin_NHS;
        end
        
        %% pruning hidden unit if 16 is satisfied
        if grow == 0 && Kg > 1 && miustd_NHS >= miustdmin_NHS && kk > I + 1 && parameter.net.mode ~= 3
            HS = Ey;                % equation 19
            [~,BB] = min(HS);       % equation 20, BB is the insignificant node
            fprintf('The node no %d is PRUNED around sample %d\n', BB, iData)
            prune       = 1;
            kprune      = kprune + 1;
            K           = K - 1;
            Kg          = Kg - 1;
            node(kk)    = K;
            nodeg(kk)   = Kg;
            b(BB)       = [];
            W(BB,:)     = [];
            miu_A_old(BB) = [];
            var_A_old(BB) = [];
            parameter.net.velocity{ly}(BB,:)            = [];
            parameter.net.grad{ly}(BB,:)                = [];
            parameter.net.weightSoftmax{ly}(:,BB+1)     = [];
            parameter.net.momentumSoftmax{ly}(:,BB+1)   = [];
            parameter.net.gradSoftmax{ly}(:,BB+1)       = [];
        else
            node(kk)  = K;
            nodeg(kk) = Kg;
            prune     = 0;
        end
        
        %% feedforward
        if grow == 1 || prune == 1
            a = W*x_tail(iData,:)' + b;     % equation 1
            h = sigmf(a,[1,0]);             % equation 1
            a_hat = W'*h + c;               % equation 2
            switch mode
                case 'sigmsigm'
                    x_hat = sigmf(a_hat,[1,0]);     % equation 2
                case 'sigmafn'
                    x_hat = a_hat;
            end
            x_hat      = x_hat';
            e(iData,:) = x(iData,:) - x_hat;
            L(iData,:) = 0.5*norm(e(iData,:))^2;
        end
    end
    
    %% Backpropaagation of DAE, tied weight
    lr      = 0.001;
    W_old   = W;
    dedxhat = -e(iData,:);
    switch mode
        case 'sigmsigm'
            del_j = x_hat.*(1 - x_hat);
        case 'sigmafn'
            del_j = ones(1,length(x_hat));
    end
    d3      = dedxhat.*del_j;
    d_act   = (h.*(1 - h))';
    d2      = (d3 * W') .* d_act;
    dW2     = (d3' * h');               % grad W of the second layer
    dW1     = (d2' * x_tail(iData,:));  % grad W of the first layer
    dW      = dW1 + dW2';               % grad W
    W       = W_old - lr*dW;            % update W, tied weight
    del_W   = del_j.*W.*d_act';         
    dedb    = dedxhat*del_W';           % grad b
    b       = b - lr*dedb';             % update b
    dejdcj  = dedxhat.*del_j;           % grad c
    c       = c - (lr*dejdcj)';         % update c
    clear dejdcj dedb del_W dW dW1 dW2 d2 d_act d3 del_j
end
a  = W*x' + b;
yh = sigmf(a,[1,0]);
h  = yh';

%% substitute the weight back to evdae
parameter.net.weight{ly} = [b W];
parameter.net.c{ly} = c;
parameter.net.K(ly) = K;
parameter.dae{ly}.node = nodeg;
parameter.ev{ly}.node = node;
parameter.dae{ly}.LF = mean(L);
parameter.dae{ly}.BIAS2 = BIAS2;
parameter.dae{ly}.VAR = VAR;
parameter.dae{ly}.kk = kk;
parameter.dae{ly}.lr = lr;
parameter.dae{ly}.K = K;
parameter.dae{ly}.Kg = Kg;
parameter.ev{ly}.K = K;
parameter.dae{ly}.miu_A_old = miu_A_old;
parameter.dae{ly}.var_A_old = var_A_old;
parameter.dae{ly}.miu_NS_old = miu_NS_old;
parameter.dae{ly}.var_NS_old = var_NS_old;
parameter.dae{ly}.miu_NHS_old = miu_NHS_old;
parameter.dae{ly}.var_NHS_old = var_NHS_old;
parameter.dae{ly}.miumin_NS = miumin_NS;
parameter.dae{ly}.miumin_NHS = miumin_NHS;
parameter.dae{ly}.stdmin_NS = stdmin_NS;
parameter.dae{ly}.stdmin_NHS = stdmin_NHS;

%% substitute the weight back to NN
disp('DEVDAN Training: ... finished');
end

%% discriminative training, similar to ADL, https://www.researchgate.net/publication/332886773_Autonomous_Deep_Learning_Continual_Learning_Approach_for_Dynamic_Environments 
function parameter  = discriminativeTraining(parameter,y,portionLabeledData,selectiveSample)
[~,bb] = size(parameter.net.weight{parameter.net.index});
grow = 0;
prune = 0;
ly = parameter.net.index;
x = parameter.net.activity{ly};
% [N,~] = size(x);

%% add adverarial samples
% if parameter.net.t > 2
%     sign_grad_Loss = sign(parameter.Loss(parameter.net.t-1) - parameter.Loss(parameter.net.t-2));
%     adversarial_samples = x + 0.007*sign_grad_Loss;
%     kk = randperm(N);
%     x = x(kk,:);
%     y1 = y(kk,:);
%
%     kk = randperm(N);
%     adversarial_samples = adversarial_samples(kk,:);
%     y2 = y(kk,:);
%     x = [x;adversarial_samples];
%     y = [y1;y2];
% end

%% initiate performance matrix
BIAS2       = parameter.ev{ly}.BIAS2;
VAR         = parameter.ev{ly}.VAR;
miu_x_old   = parameter.ev{ly}.miu_x_old;
var_x_old   = parameter.ev{ly}.var_x_old;
miu_NS_old  = parameter.ev{ly}.miu_NS_old;
var_NS_old  = parameter.ev{ly}.var_NS_old;
miu_NHS_old = parameter.ev{ly}.miu_NHS_old;
var_NHS_old = parameter.ev{ly}.var_NHS_old;
miumin_NS   = parameter.ev{ly}.miumin_NS;
miumin_NHS  = parameter.ev{ly}.miumin_NHS;
stdmin_NS   = parameter.ev{ly}.stdmin_NS;
stdmin_NHS  = parameter.ev{ly}.stdmin_NHS;
t           = parameter.ev{ly}.t;
kp          = parameter.ev{ly}.kp;
K           = parameter.ev{ly}.K;
Kd          = parameter.ev{ly}.Kd;
node        = parameter.ev{ly}.node;
noded       = parameter.ev{ly}.noded;
miu_A_old   = parameter.dae{ly}.miu_A_old;
var_A_old   = parameter.dae{ly}.var_A_old;

%% initiate training model
net = netInitTrain([1 1 1]);
net.activation_function = parameter.net.activation_function;

%% substitute the weight to be trained to training model
net.weight{1}   = parameter.net.weight{ly};
net.velocity{1} = parameter.net.velocity{ly};
net.grad{1}     = parameter.net.grad{ly};
net.weight{2}   = parameter.net.weightSoftmax{ly};
net.velocity{2} = parameter.net.momentumSoftmax{ly};
net.grad{2}     = parameter.net.gradSoftmax{ly};

%% load the data for training
[nData,I]        = size(x);
classProbability = parameter.net.activityOutput{1};
[x,y,nData]      = reduceShuffleData(x,y,nData,portionLabeledData,selectiveSample,...
    classProbability);

%% xavier initialization
n_in = parameter.net.initialConfig(1);

%% main loop, train the model
for iData = 1 : nData
    kp = kp + 1;
    
    %% Incremental calculation of x_tail mean and variance
    [miu_x,std_x,var_x] = meanstditer(miu_x_old,var_x_old,x(iData,:),kp);
    miu_x_old = miu_x;
    var_x_old = var_x;
    
    %% Expectation of z
    py = probit(miu_x,std_x);
    Ey = sigmf(net.weight{1}*py',[1,0]);
    Ey = [1;Ey];
    Ez = net.weight{2}*Ey;
    Ez = exp(Ez);
    Ez = Ez./sum(Ez);
    Ez2 = net.weight{2}*Ey.^2;
    Ez2 = exp(Ez2);
    Ez2 = Ez2./sum(Ez2);
    
    %% Network mean calculation
    bias2 = (Ez - y(iData,:)').^2;
    ns    = bias2;
    NS    = norm(ns,'fro');
    
    %% Incremental calculation of NS mean and variance
    [miu_NS,std_NS,var_NS] = meanstditer(miu_NS_old,var_NS_old,NS,kp);
    miu_NS_old = miu_NS;
    var_NS_old = var_NS;
    miustd_NS = miu_NS + std_NS;
    if kp <= 1 || grow == 1
        miumin_NS = miu_NS;
        stdmin_NS = std_NS;
    else
        if miu_NS < miumin_NS
            miumin_NS = miu_NS;
        end
        if std_NS < stdmin_NS
            stdmin_NS = std_NS;
        end
    end
    miustdmin_NS = miumin_NS + (1.3*exp(-NS)+0.7)*stdmin_NS;
    BIAS2(kp,:) = miu_NS;
    
    %% growing hidden unit
    if miustd_NS >= miustdmin_NS && kp > 1 && parameter.net.mode ~= 2
        grow = 1;
        K    = K + 1;
        Kd   = Kd + 1;
        fprintf('The new node no %d is FORMED around sample %d\n', K, iData)
        node(kp)  = K;
        noded(kp) = Kd;
        net.weight{1}   = [net.weight{1};normrnd(0,sqrt(2/(n_in+1)),[1,bb])];
        net.velocity{1} = [net.velocity{1};zeros(1,bb)];
        net.grad{1}     = [net.grad{1};zeros(1,bb)];
        net.weight{2}   = [net.weight{2} normrnd(0,sqrt(2/(K+1)),[parameter.net.initialConfig(end),1])];
        net.velocity{2} = [net.velocity{2} zeros(parameter.net.initialConfig(end),1)];
        net.grad{2}     = [net.grad{2} zeros(parameter.net.initialConfig(end),1)];
        miu_A_old       = [miu_A_old;0];
        var_A_old       = [var_A_old;0];
    else
        grow      = 0;
        node(kp)  = K;
        noded(kp) = Kd;
    end
    
    %% Network variance calculation
    var = Ez2 - Ez.^2;
    NHS = norm(var,'fro');
    
    %% Incremental calculation of NHS mean and variance
    [miu_NHS,std_NHS,var_NHS] = meanstditer(miu_NHS_old,var_NHS_old,NHS,kp);
    miu_NHS_old               = miu_NHS;
    var_NHS_old               = var_NHS;
    miustd_NHS                = miu_NHS + std_NHS;
    if kp <= I+1 || prune == 1
        miumin_NHS = miu_NHS;
        stdmin_NHS = std_NHS;
    else
        if miu_NHS < miumin_NHS
            miumin_NHS = miu_NHS;
        end
        if std_NHS < stdmin_NHS
            stdmin_NHS = std_NHS;
        end
    end
    miustdmin_NHS = miumin_NHS + 2*(1.3*exp(-NHS)+0.7)*stdmin_NHS;
    VAR(kp,:)     = miu_NHS;
    
    %% pruning hidden unit
    if grow == 0 && Kd > 0 && miustd_NHS >= miustdmin_NHS && kp > I + 1 && parameter.net.mode ~= 3
        HS      = Ey(2:end);
        [~,BB]  = min(HS);
        fprintf('The node no %d is PRUNED around sample %d\n', BB, iData)
        prune     = 1;
        K         = K - 1;
        Kd        = Kd - 1;
        node(kp)  = K;
        noded(kp) = Kd;
        net.weight{1}(BB,:)     = [];
        net.velocity{1}(BB,:)   = [];
        net.grad{1}(BB,:)       = [];
        net.weight{2}(:,BB+1)   = [];
        net.velocity{2}(:,BB+1) = [];
        net.grad{2}(:,BB+1)     = [];
        miu_A_old(BB)           = [];
        var_A_old(BB)           = [];
    else
        node(kp)    = K;
        noded(kp)   = Kd;
        prune       = 0;
    end
    
    %% feedforward
    net = netFeedForwardTrain(net, x(iData,:), y(iData,:));
    
    %% optimize the parameters
    net = lossBackward(net);        % backpropagate
    net = optimizerStep(net);       % update the weight
end

%% substitute the weight back to main model
parameter.net.weight{ly}        = net.weight{1};
parameter.net.weightSoftmax{ly} = net.weight{2};

%% reset momentumCoeff and gradient
parameter.net.velocity{ly}          = net.velocity{1}*0;
parameter.net.grad{ly}              = net.grad{1}*0;
parameter.net.momentumSoftmax{ly}   = net.velocity{2}*0;
parameter.net.gradSoftmax{ly}       = net.grad{2}*0;

%% substitute the recursive calculation
parameter.dae{ly}.miu_A_old = miu_A_old;
parameter.dae{ly}.var_A_old = var_A_old;
parameter.ev{ly}.t      = t + 1;
parameter.ev{ly}.kp     = kp;
parameter.ev{ly}.K      = K;
parameter.ev{ly}.Kd     = Kd;
parameter.ev{ly}.node   = node;
parameter.ev{ly}.noded  = noded;
parameter.ev{ly}.BIAS2  = BIAS2;
parameter.ev{ly}.VAR         = VAR;
parameter.ev{ly}.miu_x_old   = miu_x_old;
parameter.ev{ly}.var_x_old   = var_x_old;
parameter.ev{ly}.miu_NS_old  = miu_NS_old;
parameter.ev{ly}.var_NS_old  = var_NS_old;
parameter.ev{ly}.miu_NHS_old = miu_NHS_old;
parameter.ev{ly}.var_NHS_old = var_NHS_old;
parameter.ev{ly}.miumin_NS   = miumin_NS;
parameter.ev{ly}.miumin_NHS  = miumin_NHS;
parameter.ev{ly}.stdmin_NS   = stdmin_NS;
parameter.ev{ly}.stdmin_NHS  = stdmin_NHS;
end

%% reduce and shuffle data
function [inputData,targetData,nData] = reduceShuffleData(inputData,...
    targetData,nData,dataProportion,selectiveSample,classProbability)
nNewData = round(nData*dataProportion);
if selectiveSample.mode == 0
    s = RandStream('mt19937ar','Seed',0);
    ApplyPermutation = randperm(s,nData);
    inputData  = inputData(ApplyPermutation,:);
    targetData = targetData(ApplyPermutation,:);
    if dataProportion ~= 1
        noOfLabeledData = round(dataProportion*nData);
        inputData  = inputData(1:noOfLabeledData,:);
        targetData = targetData(1:noOfLabeledData,:);
    end
elseif selectiveSample.mode == 1
    nData = size(classProbability,1);
    selectedIndices = [];
    iIndices = 0;
    for iData = 1:nData
        if iIndices <= round(nData*dataProportion)
            confCandidate = sort(classProbability(iData,:),'descend');
            y1 = confCandidate(1);
            y2 = confCandidate(2);
            confFinal = y1/(y1+y2);
            if confFinal < selectiveSample.delta
                iIndices = iIndices + 1;
                selectedIndices(iIndices) = iData;
            end
        end
    end
    originalData   = inputData;
    originalTarget = targetData;
    inputData  = inputData(selectedIndices,:);
    targetData = targetData(selectedIndices,:);
    originalData(selectedIndices,:) = [];
    originalTarget(selectedIndices,:) = [];
end
[nData,~] = size(inputData);
if nData < nNewData
    nDataNeded = nNewData - nData;
    nCandidateData = size(originalData,1);
    indices = randperm(nCandidateData,nDataNeded);
    additionalData = originalData(indices,:);
    additionalTarget = originalTarget(indices,:);
    inputData = [inputData;additionalData];
    targetData = [targetData;additionalTarget];
    [nData,~] = size(inputData);
elseif nData > nNewData
    indices = randperm(nData,nNewData);
    inputData  = inputData(indices,:);
    targetData = targetData(indices,:);
    [nData,~] = size(inputData);
end
end

%% initialize network parameter
% This code aims to construct neural network with several hidden layer
% one can choose to either connect every hidden layer output to
% the last output or not
function net = netInit(layer)
net.initialConfig        = layer;                       %  initial network configuration
net.nLayer               = numel(net.initialConfig);    %  Number of layer
net.nHiddenLayer         = net.nLayer - 2;              %  number of hidden layer
net.activation_function  = 'sigmf';         %  Activation functions of hidden layers: 'sigmf', 'tanh' and 'relu'.
net.learningRate         = 0.01;            %  learning rate smaller value is preferred
net.momentumCoeff        = 0.95;            %  Momentum coefficient, higher value is preferred
net.output               = 'softmax';       %  output layer can be selected as follows: 'sigmf', 'softmax', and 'linear'

%% initiate weights and weight momentumCoeff for hidden layer
for i = 2 : net.nLayer - 1
    net.weight {i - 1}  = normrnd(0,sqrt(2/(net.initialConfig(i-1)+1)),[net.initialConfig(i),net.initialConfig(i - 1)+1]);
    net.velocity{i - 1} = zeros(size(net.weight{i - 1}));
    net.grad{i - 1}     = zeros(size(net.weight{i - 1}));
    net.c{i - 1}        = normrnd(0,sqrt(2/(net.initialConfig(i-1)+1)),[net.initialConfig(i - 1),1]);
end

%% initiate weights and weight momentumCoeff for output layer
for i = 1 : net.nHiddenLayer
    net.weightSoftmax {i}   = normrnd(0,sqrt(2/(size(net.weight{i},1)+1)),[net.initialConfig(end),net.initialConfig(i+1)+1]);
    net.momentumSoftmax{i}  = zeros(size(net.weightSoftmax{i}));
    net.gradSoftmax{i}      = zeros(size(net.weightSoftmax{i}));
    net.beta(i)             = 1;
    net.betaOld(i)          = 1;
    net.p(i)                = 1;
end
end

%% testing
function [net] = testing(net, input, trueClass, ev)
%% feedforward
net         = netFeedForward(net, input, trueClass);
[nData,~]   = size(trueClass);

%% obtain trueclass label
[~,actualLabel] = max(trueClass,[],2);

%% calculate the number of parameter
[a,b]   = size(net.weight{1});
[c,d]   = size(net.weightSoftmax{1});
nop(1)  = a*b + c*d;

%% calculate the number of node in each hidden layer
net.nodes{1}(net.t) = ev{1}.K;
net.nop(net.t)      = sum(nop) + length(net.c);
net.mnop            = [mean(net.nop) std(net.nop)];

%% calculate classification rate
[classProbability,classPerdiction] = max(net.activityOutput{1},[],2);
net.wrongPred       = find(classPerdiction ~= actualLabel);
net.classRate       = 1 - numel(net.wrongPred)/nData;
net.residual_error  = 1 - classProbability;
net.classPerdiction = classPerdiction;
net.actualLabel     = actualLabel;
end

function in = maskingnoise(in,nin,noiseIntensity)
%% input masking
if nin > 1
    if noiseIntensity > 0
        nMask    = max(round(noiseIntensity*nin),1);
        s        = RandStream('mt19937ar','Seed',0);
        mask_gen = randperm(s,nin,nMask);
        in(1,mask_gen) = 0;
    end
else
    mask_gen = rand(size(in(1,:))) > 0.3;
    in       = in*mask_gen;
end
end

%% feedforward
function net = netFeedForward(net, input, trueClass)
nLayer          = net.nLayer;
nData           = size(input,1);
input           = [ones(nData,1) input];       % by adding 1 to the first coulomn, it means the first coulomn of weight is bias
net.activity{1} = input;                       % the first activity is the input itself

%% feedforward from input layer through all the hidden layer
for iLayer = 2 : nLayer-1
    switch net.activation_function
        case 'sigmf'
            net.activity{iLayer} = sigmf(net.activity{iLayer - 1} * net.weight{iLayer - 1}',[1,0]);
        case 'tanh'
            net.activity{iLayer} = tanh(net.activity{iLayer - 1} * net.weight{iLayer - 1}');
        case 'relu'
            net.activity{iLayer} = max(net.activity{iLayer - 1} * net.weight{iLayer - 1}',0);
    end
    net.activity{iLayer} = [ones(nData,1) net.activity{iLayer}];
end

%% propagate to the output layer
for iLayer = 1 : net.nHiddenLayer
    switch net.output
        case 'sigmf'
            net.activityOutput{iLayer} = sigmf(net.activity{iLayer + 1} * net.weightSoftmax{iLayer}',[1,0]);
        case 'linear'
            net.activityOutput{iLayer} = net.activity{iLayer + 1} * net.weightSoftmax{iLayer}';
        case 'softmax'
            net.activityOutput{iLayer} = stableSoftmax(net.activity{iLayer + 1},net.weightSoftmax{iLayer});
    end
    
    %% calculate error
    net.error{iLayer} = trueClass - net.activityOutput{iLayer};
    
    %% calculate loss function
    switch net.output
        case {'sigmf', 'linear'}
            net.loss(iLayer) = 1/2 * sum(sum(net.error .^ 2)) / nData;
        case 'softmax'
            net.loss(iLayer) = -sum(sum(trueClass .* log(net.activityOutput{iLayer}))) / nData;
    end
end
end

%% stable softmax
function output = stableSoftmax(activation,weight)
output = activation * weight';
output = exp(output - max(output,[],2));
output = output./sum(output, 2);
end

%% probit function
function p = probit(miu,std)
p = (miu./(1 + pi.*(std.^2)./8).^0.5);
end

%% recursive mean and standard deviation
function [miu,std,var] = meanstditer(miu_old,var_old,x,k)
miu = miu_old + (x - miu_old)./k;
var = var_old + (x - miu_old).*(x - miu);
std = sqrt(var/k);
end

%% initialize network for training
function net = netInitTrain(layer)
net.initialConfig        = layer;
net.nLayer               = numel(net.initialConfig);
net.activation_function	 = 'sigmf';
net.learningRate         = 0.01;
net.momentumCoeff        = 0.95;
net.output               = 'softmax';
end

%% feedforward of a single hidden layer network
function net = netFeedForwardTrain(net, input, trueClass)
nLayer = net.nLayer;
nData  = size(input,1);
net.activity{1} = input;

%% feedforward from input layer through all the hidden layer
for iLayer = 2 : nLayer-1
    switch net.activation_function
        case 'sigmf'
            net.activity{iLayer} = sigmf(net.activity{iLayer - 1} * net.weight{iLayer - 1}',[1,0]);
        case 'relu'
            net.activity{iLayer} = max(net.activity{iLayer - 1} * net.weight{iLayer - 1}',0);
        case 'tanh'
            net.activity{iLayer} = tanh(net.activity{iLayer - 1} * net.weight{iLayer - 1}');
    end
    net.activity{iLayer} = [ones(nData,1) net.activity{iLayer}];
end

%% propagate to the output layer
switch net.output
    case 'sigmf'
        net.activity{nLayer} = sigmf(net.activity{nLayer - 1} * net.weight{nLayer - 1}',[1,0]);
    case 'linear'
        net.activity{nLayer} = net.activity{nLayer - 1} * net.weight{nLayer - 1}';
    case 'softmax'
        net.activity{nLayer} = stableSoftmax(net.activity{nLayer - 1},net.weight{nLayer - 1});
end

%% calculate error
net.error = trueClass - net.activity{nLayer};
end

%% backpropagation
function net = lossBackward(net)
nLayer = net.nLayer;
switch net.output
    case 'sigmf'
        backPropSignal{nLayer} = - net.error .* (net.activity{nLayer} .* (1 - net.activity{nLayer}));
    case {'softmax','linear'}
        backPropSignal{nLayer} = - net.error;          % loss derivative w.r.t. output
end

for iLayer = (nLayer - 1) : -1 : 2
    switch net.activation_function
        case 'sigmf'
            actFuncDerivative = net.activity{iLayer} .* (1 - net.activity{iLayer}); % contains b
        case 'tanh'
            actFuncDerivative = 1 - net.activity{iLayer}.^2;
        case 'relu'
            actFuncDerivative = zeros(1,length(net.activity{iLayer}));
            actFuncDerivative(net.activity{iLayer}>0) = 0.1;
    end
    
    if iLayer+1 == nLayer
        backPropSignal{iLayer} = (backPropSignal{iLayer + 1} * net.weight{iLayer}) .* actFuncDerivative;
    else
        backPropSignal{iLayer} = (backPropSignal{iLayer + 1}(:,2:end) * net.weight{iLayer}) .* actFuncDerivative;
    end
end

for iLayer = 1 : (nLayer - 1)
    if iLayer + 1 == nLayer
        net.grad{iLayer} = (backPropSignal{iLayer + 1}' * net.activity{iLayer});
    else
        net.grad{iLayer} = (backPropSignal{iLayer + 1}(:,2:end)' * net.activity{iLayer});
    end
end
end

%% weight update
function net = optimizerStep(net)
for iLayer = 1 : (net.nLayer - 1)
    grad                    = net.grad{iLayer};
    net.velocity{iLayer}    = net.momentumCoeff*net.velocity{iLayer} + net.learningRate * grad;
    finalGrad               = net.velocity{iLayer};
    
    %% apply the gradient to the weight
    net.weight{iLayer}      = net.weight{iLayer} - finalGrad;
end
end

%% Performance measure
% This function is developed from Gregory Ditzler
% https://github.com/gditzler/IncrementalLearning/blob/master/src/stats.m
function [fMeasure,gMean,recall,precision,error] = performanceMeasure(trueClass, rawOutput, nClass)
label           = index2vector(trueClass, nClass);
predictedLabel  = index2vector(rawOutput, nClass);

recall      = calculate_recall(label, predictedLabel, nClass);
error       = 1 - sum(diag(predictedLabel'*label))/sum(sum(predictedLabel'*label));
precision   = calculate_precision(label, predictedLabel, nClass);
gMean       = calculate_g_mean(recall, nClass);
fMeasure    = calculate_f_measure(label, predictedLabel, nClass);


    function gMean = calculate_g_mean(recall, nClass)
        gMean = (prod(recall))^(1/nClass);
    end

    function fMeasure = calculate_f_measure(label, predictedLabel, nClass)
        fMeasure = zeros(1, nClass);
        for iClass = 1:nClass
            fMeasure(iClass) = 2*label(:, iClass)'*predictedLabel(:, iClass)/(sum(predictedLabel(:, iClass)) + sum(label(:, iClass)));
        end
        fMeasure(isnan(fMeasure)) = 1;
    end

    function precision = calculate_precision(label, predictedLabel, nClass)
        precision = zeros(1, nClass);
        for iClass = 1:nClass
            precision(iClass) = label(:, iClass)'*predictedLabel(:, iClass)/sum(predictedLabel(:, iClass));
        end
        precision(isnan(precision)) = 1;
    end

    function recall = calculate_recall(label, predictedLabel, nClass)
        recall = zeros(1, nClass);
        for iClass = 1:nClass
            recall(iClass) = label(:, iClass)'*predictedLabel(:, iClass)/sum(label(:, iClass));
        end
        recall(isnan(recall)) = 1;
    end

    function output = index2vector(input, nClass)
        output = zeros(numel(input), nClass);
        for iData = 1:numel(input)
            output(iData, input(iData)) = 1;
        end
    end
end
