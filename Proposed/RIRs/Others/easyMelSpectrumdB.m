function X_mel = easyMelSpectrumdB(sig, fs)

% [sig,fs] = audioread(wavName);

nfft = 1024;
n_mels = 80;
hopSize = 256;

fmin = 125;
fmax = 7600;

X_linear = stft(sig, nfft, nfft, hopSize);
[w,cf]=filtbankm(n_mels,nfft,fs,fmin,fmax,'m');
% figure;plot(w(1:5:end,:)')

% The mel filterbanks needs normalisation with unit area
f = linspace(0,fs/2,nfft/2+1);
invarea = 2 ./ trapz(f,w');
w_norm = repmat(invarea(:),[1,nfft/2+1]).*w;
% figure;plot(w_norm(1:5:end,:)')

X_mel = 20*log10(w_norm*abs(X_linear));
% figure;
% pcolor(X_mel);shading interp;colormap(jet);colorbar;%caxis([-pi pi]);
