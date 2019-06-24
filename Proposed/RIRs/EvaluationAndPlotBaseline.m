% Apply evaluations to separated signals.
clear all; close all; clc


randn('seed',123456789);
rand('seed',123456789);

data_dir = '/vol/vssp/AP_datasets/audio/S3A_BSSOIMs/TrainEncoder/EvaluationResults/';
data_dir2 = '/user/HS204/ql0002/Desktop/speech-denoising-wavenet-master/EvaluationResults/';


SampleN = 100;



PESQResults = zeros(1,SampleN);
SIResults = zeros(1,SampleN);
SDRResults = zeros(1,SampleN);

for sample_i = 1:SampleN
    
    sourceName = sprintf('Ind_%d_source.wav',sample_i-1);
    mixName = sprintf('Ind_%d_mixture.wav',sample_i-1);
    BaselineName = sprintf('Ind_%d_est_wavenet_denoised.wav',sample_i-1);
    noiseName = sprintf('Ind_0_est_wavenet_noise.wav',sample_i-1);
    
    [source,Fs] = audioread([data_dir,sourceName]);
    [mixture,Fs] = audioread([data_dir,mixName]);
    [Estimate_baseline,Fs] = audioread([data_dir2,BaselineName]);
    [Estimate_noise,Fs] = audioread([data_dir2,noiseName]);
    
    %%%%% model.half_receptive_field_length = 3069
    source = source(3069:3069+length(Estimate_baseline)-1);
    mixture = mixture(3069:3069+length(Estimate_baseline)-1);
    
    
    %         figure; starti = Fs*2.08;endi = Fs*2.25;
    %         plot(source(starti:endi));
    %         hold on;
    %         %         syn_mix = 0;
    %         %         plot(mixture(starti+syn_mix:endi+syn_mix));
    %         syn_1 = 0;
    %         plot(Estimate_baseline(starti+syn_1:endi+syn_1));
    %         plot(Estimate_noise(starti+syn_1:endi+syn_1));
    %         a = 0;
    
    estimate = Estimate_baseline;
    
    
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



pesq_mean = mean(PESQResults(:));
si_mean = mean(SIResults(:));
fprintf('The average PESQ and STOI values are  %f-------   %f\n',pesq_mean,si_mean);

sdr_mean = mean(SDRResults(:));
fprintf('The average SDR values is  %f-------   %f\n',sdr_mean);


