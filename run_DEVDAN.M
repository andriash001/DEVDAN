clc
% clear
% close all
tic

%% load data
% load('C:\Users\andri\Documents\cifar\cifar.mat'); I = 1024;
% load('C:\Users\andri\Documents\mnist\datamnist.mat'); I = 784;
% load('C:\Users\andri\Documents\mnist\permutedmnist3task.mat'); I = 784;
% load('C:\Users\andri\Documents\mnist\14RMnist.mat'); I = 784;
% load('C:\Users\andri\Documents\SUSY\susytraintest.mat'); I = 18; %data = data(1:2000000,:);
% load('C:\Users\andri\Documents\SUSY\SUSYprocess.mat'); I = 7; %data = data(1:2000000,:);
% load('C:\Users\andri\Documents\RLCPS\rlcps.mat'); I = 9; %data = data(1:2000000,:);
% load('C:\Users\andri\Documents\KDD\kdd10persenshuff3.mat'); I = 41;
% load('C:\Users\andri\Documents\HIGGS\higgs.mat'); I = 28; %data = data(1:2000000,:);
% load('C:\Users\andri\Documents\Hepmass\Hepmass.mat'); I = 28; data = data(1:2000000,:);
% load('C:\Users\andri\Documents\PokerHand\pokertrts1000fold.mat'); I = 10;
% load('C:\Users\andri\Documents\rfid\rfid_new.mat');I = 3;data(:,3) = [];
% load('C:\Users\andri\Documents\rfid\rfid_ref.mat');I = 1;

% Available in this folder
% load weather ; I = 8;
% load rfid; I = 3;
% load electricitypricing; I = 8;
% load hyperplane; I = 4;
load sea; I = 3;
% load occupancy; I = 5;
% load('heart_disease_normalized.mat'); I = 13;

%% This is ONLY for the numerical example in subsectio 4.9
selectiveSample.mode  = 0;       % 0: selective sample off, 1: selective sample on
selectiveSample.delta = 0.7;     % confidence level; 
portionOfLabeledData  = 1;       % 0-1

%% DEVDAN
chunkSize = 500;                % number of data in a batch
mode      = 0; %0: all components are on, 1: generative off, 2: growing hidden unit off, 3: pruning hidden unit off
[parameter,performance] = DEVDAN(data,I,portionOfLabeledData,mode,...
    selectiveSample,chunkSize);
clear data
disp(performance)