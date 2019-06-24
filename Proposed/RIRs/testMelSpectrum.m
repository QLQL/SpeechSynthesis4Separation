clear all; close all; clc
% train the model for clean binaural speech signals
load('G57_mic_2205.mat')
Azimuth_array = [0, 15, -15, 30, -30, 60, -60, 90, -90];

wavName = sprintf('/vol/vssp/ucdatasets/s3a/Qingju/LJSpeech-1.1/wavs/LJ%03d-%04d.wav',5,16);

[sig,fs] = audioread(wavName);

X_mel = easyMelSpectrumdB(sig, fs);
figure;
pcolor(X_mel);shading interp;colormap(jet);colorbar;%caxis([-pi pi]);


targetName = sprintf('/vol/vssp/ucdatasets/s3a/Qingju/LJSpeech-1.1/wavs/LJ%03d-%04d.wav',6,16);
InterfName = sprintf('/vol/vssp/datasets/audio/S3A_BSSOIMs/TSP/48k/FH/FH46_01.wav')

h1 = squeeze(ir(1,1:2,:)); h1 = h1';
h2 = squeeze(ir(6,1:2,:)); h2 = h2';

[s1_origin, fs] = audioread(targetName);
[s2_origin, Fs] = audioread(InterfName);
s2_origin = resample(s2_origin,fs,Fs);
% figure;plot(s1_origin(1:fs*1));hold on;plot(s2_origin(1:fs*1));hold off;

L = max(length(s1_origin),length(s2_origin));
%                     timeSum = timeSum+L/Fs;
Ind = mod([1:L], length(s1_origin)); Ind(Ind==0) = length(s1_origin);
s1_origin = s1_origin(Ind);
Ind = mod([1:L], length(s2_origin)); Ind(Ind==0) = length(s2_origin);
s2_origin = s2_origin(Ind);

% generate the convolutive mixtures
lr1 = [];lr2 = [];
lr1(:,1) = fftfilt(h1(:,1),s1_origin);
lr1(:,2) = fftfilt(h1(:,2),s1_origin);
lr2(:,1) = fftfilt(h2(:,1),s2_origin);
lr2(:,2) = fftfilt(h2(:,2),s2_origin);

lr = lr1+lr2;
scale = 1/max(abs(lr(:)))*0.99;
lr = lr*scale;
lr1 = lr1*scale;
lr2= lr2*scale;

lr1_mel = easyMelSpectrumdB(mean(lr1,2), fs);

lr2_mel = easyMelSpectrumdB(mean(lr2,2), fs);

lr_mel = easyMelSpectrumdB(mean(lr,2), fs);

figure;
subplot(311);pcolor(lr1_mel);shading interp;colormap(jet);colorbar;%caxis([-pi pi]);
subplot(312);pcolor(lr2_mel);shading interp;colormap(jet);colorbar;
subplot(313);pcolor(lr_mel);shading interp;colormap(jet);colorbar;

% t = 100;figure;plot([lr1_mel(:,t),lr2_mel(:,t),lr_mel(:,t)],'*-')

lr_mel_max = max(lr1_mel,lr2_mel);

lr_diff = lr_mel-lr_mel_max;
figure;hist(lr_mel(:),100);
figure;hist(lr_diff(:),100)

threshold = 2;
figure;
index = abs(lr_diff)<=threshold;
scatter(lr1_mel(index),lr2_mel(index),'*')
index = abs(lr_diff)>threshold;
hold on;
scatter(lr1_mel(index),lr2_mel(index),'*')
plot([-80 -80], [20 20])
hold off;

a=0