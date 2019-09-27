% LicenseCC BY-NC-SA 4.0
% 
% Copyright (c) 2018 Andri Ashfahani Mahardhika Pratama

clc
clear
close all

% All the datasets used in our experiments can be downloaded in this link: https://bit.ly/2mhtRsE
%% load data
% load rotatedmnist; 	I = 784;
% load permutedmnist; 	I = 784;
% load mnist; 		    I = 784;
% load forestcovtype; 	I = 54;
% load sea; 		    I = 3;
% load hyperplane;  	I = 4;
% load occupancy; 	    I = 7;
% load kddcup; 		    I = 41;
% load Hepemass; 	    I = 27;

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
