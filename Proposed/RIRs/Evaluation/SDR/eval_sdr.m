function SDR=eval_sdr(Est_s1,origin,fs)

% Est_s1 is the estimated signal (Lx1)
% origin is the time history (N) of the target signal before mixing
% The output is the Signal-to-Distortion ratio

if nargin<3,
    fs=8000;
end
% it is the clean utterance
s1=origin(1:length(Est_s1));% Because reconstruct function changes the length

%q=15;
% q=round(0.032*fs); % a Wiener filter of 32 ms
q=256;

% Any energy in the estimated signal that can be explained with a 
% linear combination of delayed versions of the target signal
h_wien=wiener_filter2(s1,Est_s1,q);
S1_hat=filter(h_wien,1,s1);

target_energy=var(S1_hat);
distortion_energy=var(Est_s1-S1_hat);

SDR=10*log10(target_energy/distortion_energy);

