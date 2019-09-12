clc
clear
close all


%% load data
% load hyperplane; I = 4;
load sea; I = 3;

%% This is ONLY for the numerical example where number of labeled data is limited, section 4.9
selectiveSample.mode  = 0;       % 0: selective sample off, 1: selective sample on
selectiveSample.delta = 0.55;    % confidence level; 
portionOfLabeledData  = 1;       % 0-1

%% DEVDAN
chunkSize = 500;        % number of data in a batch
mode      = 0;          % 0: all components are on, 1: generative off, 2: growing hidden unit off, 3: pruning hidden unit off
[parameter,performance] = DEVDAN(data,I,portionOfLabeledData,mode,...
    selectiveSample,chunkSize);
clear data
disp(performance)