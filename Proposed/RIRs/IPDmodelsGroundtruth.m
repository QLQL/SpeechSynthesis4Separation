% Where the stereo signals contains only one sound source from a specific
% direction, we calculate the groundtruth IPD mean and variance

clear all; close all; clc
% train the model for clean binaural speech signals
load('G57_mic_2205.mat')
Azimuth_array = [0, 15, -15, 30, -30, 60, -60, 90, -90];

% concatnate a group of LJSpeech sequences 
sig = [];
Fs = 22050; % Hz
randn('seed',111111111111111111);
rand('seed',111111111111111111);
for i = 1:5,
    ind1 = randi(50,1);
    ind2 = randi(100,1);
    wavName = sprintf('/vol/vssp/ucdatasets/s3a/Qingju/LJSpeech-1.1/wavs/LJ%03d-%04d.wav',ind1,ind2);
    temp = audioread(wavName);
    sig = [sig,temp'];
end



verbose = 1;

NFFT = 1024;
Window = hanning(NFFT);
Shift = 0.25;
startf = round(300/Fs*NFFT);
endf = round(3400/Fs*NFFT);

IPD_mean = zeros(NFFT/2+1,length(Azimuth_array)); %%%%%
IPD_var = zeros(NFFT/2+1,length(Azimuth_array));%%%%%
for azi_iii = 1:length(Azimuth_array),
    azi = Azimuth_array(azi_iii);
    fprintf('\n The azimuth of the speaker is %d +++++++++++++++++++\n', azi);
    h1 = squeeze(ir(azi_iii,:,:));
    h1 = h1';
    
    
    % generate the mixtures
    l = fftfilt(h1(:,1),sig);
    r = fftfilt(h1(:,2),sig);
    L = stft(l, NFFT, Window, Shift);
    R = stft(r, NFFT, Window, Shift);
    
    TT = size(L,2);
    
    neuma=(L).*conj(R);
    deno=abs(neuma)+eps;
    GPHAT=neuma./deno;
    %GPHAT = GPHAT(2:end-1,:);
    
    %     % The following is equivalent to the above
    %     GPHAT = exp(1i*(angle(L)-angle(R)));
    
    ang = angle(L./R);
    
    %     tau = -12:0.5:12;% Fs*real_tau_array
    %     w = [0:NFFT/2]' * 2*pi / NFFT; %(0 pi), 2*pi*f/Fs %%%%%
    tau = linspace(-0.001, 0.001, 41);
    f_array = linspace(0, Fs / 2, NFFT/2+1);
    w = 2*pi*f_array(:);
    expwtau = exp(-1i * w * tau);
    
    T = sum(real(expwtau(startf:endf,:).'* GPHAT(startf:endf,:)),2);
    [~,opt] = max(T);
    expwtau_opt = exp(-1i * w * tau(opt));
    
    angle_orig = -angle(expwtau_opt);
    angle_current = angle_orig;
    
    if verbose,
        figure('position',[260*(azi_iii-1) 100 300 800]);subplot(411);pcolor(ang');shading interp;caxis([-pi pi]);colormap(jet);%colorbar;
        subplot(412);plot(angle_current); axis tight; ylim([-pi, pi]);
        
        residue = angle(repmat(exp(-1i * angle_current),[1 TT]).*GPHAT);
        subplot(413);pcolor(residue');shading interp;caxis([-pi pi]);colorbar;colormap(jet)
    end
    
    for ii = 1:5,
        residue = angle(repmat(exp(-1i * angle_current),[1 TT]).*GPHAT);
        angleIncrement = mean(residue,2);
        angle_current = angle_current+angleIncrement;
        angle_current(angle_current>pi) = angle_current(angle_current>pi)-2*pi;
        angle_current(angle_current<-pi) = angle_current(angle_current<-pi)+2*pi;
        
        if verbose,
            subplot(413);pcolor(residue');shading interp;caxis([-pi pi]);colorbar;colormap(jet)
            subplot(414);plot(angleIncrement);hold on; plot(angle_current);hold off
        end
        
    end
    % calculate the variance
    residue = angle(repmat(exp(-1i * angle_current),[1 TT]).*GPHAT);
    var_current = var(residue,0,2);
%     if verbose,
%         figure;
%         plot(angle_current+sqrt(var_current(:)),'k--');
%         plot(angle_current-sqrt(var_current(:)),'k--');
%     end
    
    IPD_mean(:,azi_iii) = angle_current;
    IPD_var(:,azi_iii) = var_current;
end

figure;plot(IPD_mean)
figure;plot(sqrt(IPD_var))
saveName = 'tempIPDParams.mat'
% save(saveName,'IPD_mean','IPD_var','Azimuth_array')
