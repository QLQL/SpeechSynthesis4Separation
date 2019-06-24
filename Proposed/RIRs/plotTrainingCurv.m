clear all; close all; clc

filename1 = 'log_mel.csv';
M1 = readtable(filename1);

filename2 = 'log_linear.csv';
M2 = readtable(filename2);




temp1 = table2array(M1(:,'my_loss'));
temp2 = table2array(M2(:,'my_loss'));

temp1_val = table2array(M1(:,'val_my_loss'));
temp2_val = table2array(M2(:,'val_my_loss'));


figure('Position', [600, 400, 400, 200]);
hold on; %yyaxis left
plot(temp1(1:500),'linewidth',2);
plot(temp1_val(1:500));
% ylabel('Loss-proposed','fontsize',12)
ylim([0 0.02])
%yyaxis right
plot(temp2(1:500),'linewidth',2)
plot(temp2_val(1:500));
legend({'Proposed-train','Proposed-val','Baseline-train','Baseline-val'});
ylabel('Loss-proposed','fontsize',12)

% temp1 = table2array(M1(:,'val_my_loss'));
% temp2 = table2array(M2(:,'val_my_loss'));
% 
% yyaxis right
% hold on;
% plot(temp1(1:500),'linewidth',2);
% plot(temp2(1:500),'linewidth',2)

a = 0
