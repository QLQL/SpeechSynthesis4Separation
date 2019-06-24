clc
clear
clear all

fid = fopen('C:/Database277/wav/test_files.l', 'rt');

while feof(fid) == 0
    tline=fgetl(fid);
    tline
    root='C:/Database277/wav/wav_mono_8000/';
   [noisy_speech, fs, nbits]= wavread( [root,tline,'.wav']);
    [vad, logVAD]= apply_VAD(noisy_speech,300);
    vad=vad';
    write_htk(strcat('C:\Database277\wav\ss_rdcVAD\clean\',tline,'.vad'),vad, size(vad,1), 100000, 4*size(vad,2), 9);
    
   
end
fclose(fid);
