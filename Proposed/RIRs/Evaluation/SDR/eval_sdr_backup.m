function SDR=eval_sdr(Est_s1,origin)

% Est_s1 is the estimated signal (Lx2)
% origin is the time history (N) of the target signal before mixing
% The output is the Signal-to-Distortion ratio

% Modified on 04/11/2010 15:17 by Atiyeh.A

% it is the clean utterance from TIMIT
s1=origin(1:length(Est_s1));% Because reconstruct function changes the length

%s2=interfer(1:length(Est_s1));% Because reconstruct function changes the length

fs=16000;
%q=15;
q=0.032*fs; % a Wiener filter of 32 ms

Estim_s1=sum(Est_s1,2);

% To check the code
%[s_target,e_interf,e_artif]=bss_decomp_filt(estim_s1',1,[s1';s2'],q);
%[SDR,SIR,SAR]=bss_crit(s_target,e_interf,e_artif);

% Any energy in the estimated signal that can be explained with a 
% linear combination of delayed versions of the target signal
% h_wien=wiener_filter2(Estim_s1,s1,q);
h_wien=wiener_filter2(s1,Estim_s1,q);
S1_hat=filter(h_wien,1,s1);

target_energy=var(S1_hat);
distortion_energy=var(Estim_s1-S1_hat);

SDR=10*log10(target_energy/distortion_energy);

