function d = stft(x, NFFT, Window_or_WindowSize, HopSize_or_HopPercent)
% Short-time Fourier transform.
% d = stft(x, NFFT, Window_or_WindowSize, HopSize_or_HopPercent)
% NFFT should be a number power to 2.
% Qingju Liu

if nargin<2, NFFT = 1024; end
if nargin<4, HopSize_or_HopPercent=.25; end
if nargin<3, Window_or_WindowSize = NFFT;end

if length(Window_or_WindowSize)==1,
 	Window=hamming(Window_or_WindowSize);
else
 	Window=Window_or_WindowSize(:);
end
W = length(Window);

if HopSize_or_HopPercent>1,
	HopSize = round(HopSize_or_HopPercent);
	HopPercent = HopSize/W;
else
	HopPercent = HopSize_or_HopPercent;
	HopSize = fix(W.*HopSize_or_HopPercent);
end


Seg = segment(x,W,HopPercent,Window);
SegFFT = fft(Seg,NFFT);
d = SegFFT(1:(NFFT/2+1),:);

