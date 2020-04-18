% (2X2) VBLAST - MIMO OFDM
clear all
close all
clc
num_frames = 10^1; % simulation runs
FFT_len = 1024; % length of the FFT/IFFT
chan_len = 10; % number of channel taps
cp_len = chan_len-1; % length of the cyclic prefix
fade_var_1D = 0.5; % 1D fade variance

% SNR parameters
SNR_dB = 40; % SNR per bit (dB)
% AWGN variance per dimension per receive diversity arm
noise_var_1D = 2*fade_var_1D*chan_len*2*2*2/(2*10^(0.1*SNR_dB)*FFT_len*2*2);

C_Ber = 0; % bit errors in each frame
tic()
for frame_cnt = 1:num_frames
%---------------------------TRANSMITTER------------------------------------
% QPSK symbols from transmit antenna 1
a1 = randi([0 1],1,2*FFT_len);
qpsk_seq1 = 1-2*a1(1:2:end) + 1i*(1-2*a1(2:2:end));

% QPSK symbols from transmit antenna 2
a2 = randi([0 1],1,2*FFT_len);
qpsk_seq2 = 1-2*a2(1:2:end) + 1i*(1-2*a2(2:2:end));

% IFFT operation at transmitter 1
T_sig1 = ifft(qpsk_seq1);

% IFFT operation at transmitter 2
T_sig2 = ifft(qpsk_seq2);

% Insert cyclic prefix at transmitter 1
T_sig_CP1 = [T_sig1(end-cp_len+1:end) T_sig1];

% Insert cyclic prefix at transmitter 2
T_sig_CP2 = [T_sig2(end-cp_len+1:end) T_sig2];

% -------------------- CHANNEL --------------------------------------------
% fade channel for transmitter 1 and receiver 1
fade_chan11 = normrnd(0,sqrt(fade_var_1D),1,chan_len)+1i*...
    normrnd(0,sqrt(fade_var_1D),1,chan_len);
fade_chan_FFT11 = fft(fade_chan11,FFT_len);

% fade channel for transmitter 2 and receiver 1
fade_chan21 = normrnd(0,sqrt(fade_var_1D),1,chan_len)+1i*...
    normrnd(0,sqrt(fade_var_1D),1,chan_len);
fade_chan_FFT21 = fft(fade_chan21,FFT_len);


% fade channel for transmitter 1 and receiver 2
fade_chan12 = normrnd(0,sqrt(fade_var_1D),1,chan_len)+1i*...
    normrnd(0,sqrt(fade_var_1D),1,chan_len);
fade_chan_FFT12 = fft(fade_chan12,FFT_len);

% fade channel for transmitter 2 and receiver 2
fade_chan22 = normrnd(0,sqrt(fade_var_1D),1,chan_len)+1i*...
    normrnd(0,sqrt(fade_var_1D),1,chan_len);
fade_chan_FFT22 = fft(fade_chan22,FFT_len);


% AWGN at receiver 1
noise1 = normrnd(0,sqrt(noise_var_1D),1,FFT_len+cp_len+chan_len-1)+1i*...
    normrnd(0,sqrt(noise_var_1D),1,FFT_len+cp_len+chan_len-1);

% AWGN at receiver 2
noise2 = normrnd(0,sqrt(noise_var_1D),1,FFT_len+cp_len+chan_len-1)+1i*...
    normrnd(0,sqrt(noise_var_1D),1,FFT_len+cp_len+chan_len-1);

% Get channel output for first receiver (diversity arm)
chan_op1 = conv(fade_chan11,T_sig_CP1)+conv(fade_chan21,T_sig_CP2)+noise1;

% Get channel output for second receiver (diversity arm)
chan_op2 = conv(fade_chan12,T_sig_CP1)+conv(fade_chan22,T_sig_CP2)+noise2;

%------------- VBLAST RECEIVER --------------------------------------------
start_inx = cp_len+1; % starting index
end_inx = start_inx + FFT_len-1; % end index

% FFT output at diversity arm 1
Rx_FFT_op1 = fft(chan_op1(start_inx:end_inx));

% FFT output at diversity arm 2
Rx_FFT_op2 = fft(chan_op2(start_inx:end_inx));

dec_a1 = zeros(1,2*FFT_len);
dec_a2 = zeros(1,2*FFT_len);
for i1 = 1:FFT_len
   H_matrix = [fade_chan_FFT11(i1) fade_chan_FFT21(i1);...
       fade_chan_FFT12(i1) fade_chan_FFT22(i1)];
   pseudo_inv_H = pinv(H_matrix);
   % step 1
   y1 = pseudo_inv_H(1,:)*[Rx_FFT_op1(i1);Rx_FFT_op2(i1)] ;
   % ML decision
   dec_a1(i1*2-1)= real(y1)<0;
   dec_a1(i1*2)= imag(y1)<0;
   
   % step 2
   y2 = [Rx_FFT_op1(i1);Rx_FFT_op2(i1)] - H_matrix(:,1)*(1-2*dec_a1(i1*2-1)+...
       1i*(1-2*dec_a1(i1*2))); 
   % performing Maximum Ration Combining
   h=H_matrix(:,2);
   h=h/norm(h); % beamformer
   y2 = h'*y2;
   % ML decision
   dec_a2(i1*2-1)= real(y2)<0;
   dec_a2(i1*2)= imag(y2)<0;
end

% BIT ERRORS
C_Ber = C_Ber+ nnz(a1-dec_a1+a2-dec_a2);
end
toc()
%BIT ERROR RATE
BER = C_Ber/(num_frames*(4*FFT_len))