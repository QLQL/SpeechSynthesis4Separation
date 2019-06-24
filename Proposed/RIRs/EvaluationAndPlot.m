% Apply evaluations to separated signals.
clear all; close all; clc


randn('seed',123456789);
rand('seed',123456789);

% data_dir = '/vol/vssp/AP_datasets/audio/S3A_BSSOIMs/TrainEncoder/NewResultsSamples/';
data_dir = '/vol/vssp/AP_datasets/audio/S3A_BSSOIMs/TrainEncoder/EvaluationResults/';


SampleN = 100;
AngN = 4;
for algorithm = 1,%1:2,
    
    PESQResults = zeros(1,SampleN);
    SIResults = zeros(1,SampleN);
    SDRResults = zeros(1,SampleN);
    
    for sample_i = 1:SampleN
        
        sourceName = sprintf('Ind_%d_source.wav',sample_i-1);
        mixName = sprintf('Ind_%d_mixture.wav',sample_i-1);
        MelName = sprintf('Ind_%d_est_mel.wav',sample_i-1); % gtWaveNet performing wavenet synthesis on groundtruth Mel spectrum
        linearName = sprintf('Ind_%d_est_linear2.wav',sample_i-1); % linear2 is with groundtruth phase information
        IBMName = sprintf('Ind_%d_est_IBM.wav',sample_i-1);
        
        [source,Fs] = audioread([data_dir,sourceName]);
        [mixture,Fs] = audioread([data_dir,mixName]);
        [Estimate_mel,Fs] = audioread([data_dir,MelName]);
        Estimate_mel = Estimate_mel(1:length(source));
        [Estimate_linear,Fs] = audioread([data_dir,linearName]);
        Estimate_linear = Estimate_linear(1:length(source));
        [Estimate_IBM,Fs] = audioread([data_dir,IBMName]);
        Estimate_IBM = Estimate_IBM(1:length(source));
        
        %         figure; starti = Fs*2.08;endi = Fs*2.25;
        %         plot(source(starti:endi));
        %         hold on;
        % %         syn_mix = 0;
        % %         plot(mixture(starti+syn_mix:endi+syn_mix));
        % %         syn_mel = -128;
        % %         plot(Estimate_mel(starti+syn_mel:endi+syn_mel));
        % %         syn_lin = 0;
        % %         plot(Estimate_linear(starti+syn_lin:endi+syn_lin));
        %         syn_IBM = 0;
        %         plot(Estimate_IBM(starti+syn_IBM:endi+syn_IBM));
        
        % do the synchronisation
        %         syn_mix = 80;
        %         syn_mel = -40;
        %         mixture = mixture(1-syn_mel+syn_mix:end,1);
        %         source = source(1-syn_mel:end-syn_mix);
        %         Estimate_linear = Estimate_linear(1-syn_mel+syn_mix:-syn_mel+syn_mix+length(source));
        %         Estimate_mel = Estimate_mel(1:length(source));
        
        %             figure;
        %             starti = Fs*1;endi = Fs*1.2;
        %             plot(source(starti:endi));
        %             hold on;
        %             plot(mixture(starti:endi));
        %             plot(Estimate_mel(starti:endi));
        %             plot(Estimate_linear(starti:endi));
        
        
        if algorithm==1,
            estimate = Estimate_mel;
            %             estimate = mixture;
            %             estimate = Estimate_IBM;
            %             estimate = Estimate_linear;
        elseif algorithm==2,
            estimate = Estimate_linear;
        end
        
        %% SI score
        try
            stoival = stoi(source,estimate,Fs);
        catch
            stoival = -100;
            disp('The current STOI calculation failed');
        end
        
        SIResults(sample_i) = stoival;
        %             fprintf('The current STOI is %f\n',stoival);
        
        
        
        %% PESQ score
        try
            estimate_sub = resample(estimate,16000,Fs);
            source_sub = resample(source,16000,Fs);
            pesqval = pesq(estimate_sub,source_sub,16000);
        catch
            pesqval = -100;
            disp('The current PESQ calculation failed');
        end
        PESQResults(sample_i) = pesqval;
        %             fprintf('The current PESQ is %f \n',pesqval);
        
        
        fprintf('The current STOI-------PESQ values are  %f-------   %f\n',stoival,pesqval);
        
        
        %% SDR score
        try
            sdrval=eval_sdr(estimate,source,Fs);
        catch
            sdrval = -100;
            disp('The current SDR calculation failed');
        end
        SDRResults(sample_i) = sdrval;
        fprintf('The current SDR is %f \n',sdrval);
    end
    
    if algorithm==1,
        PESQResults_mel = PESQResults;
        SIResults_mel = SIResults;
        SDRResults_mel = SDRResults;
    elseif algorithm==2,
        PESQResults_linear = PESQResults;
        SIResults_linear = SIResults;
        SDRResults_linear = SDRResults;
    end
    
    pesq_mean = mean(PESQResults(:));
    si_mean = mean(SIResults(:));
    fprintf('The average PESQ and STOI values are  %f-------   %f\n',pesq_mean,si_mean);
    
    sdr_mean = mean(SDRResults(:));
    fprintf('The average SDR values is  %f-------   %f\n',sdr_mean);
    
end

pesq_mel_mean = mean(PESQResults_mel(:));
pesq_linear_mean = mean(PESQResults_linear(:));

fprintf('The average PESQ values for MEL and Linear are  %f-------   %f\n',pesq_mel_mean,pesq_linear_mean);

si_mel_mean = mean(SIResults_mel(:));
si_linear_mean = mean(SIResults_linear(:));

fprintf('The average STOI values for MEL and Linear are  %f-------   %f\n',si_mel_mean,si_linear_mean);

% sdr_mel_mean = mean(SDRResults_mel(:));
% sdr_linear_mean = mean(SDRResults_linear(:));
%
% fprintf('The average SDR values for MEL and Linear are  %f-------%f\n',sdr_mel_mean,sdr_linear_mean);

% save(['Results.mat'],'PESQResults_mel','SIResults_mel', 'SDRResults_mel','PESQResults_linear','SIResults_linear','SDRResults_linear')
% save(['Results.mat'],'PESQResults_mel','SIResults_mel','PESQResults_linear','SIResults_linear')

% load('MelResults.mat')
% mean(PESQResults_mel)
% mean(PESQResults_linear)
% mean(SIResults_mel)
% mean(SIResults_linear)

