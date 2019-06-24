clear all; close all; clc

load PlotResult_mel.mat

figure('Position', [200, 100, 300, 600]);

subplot(311);
pcolor(mix_spec_mel');
shading interp;
% set(gca,'XTick',[]);
% set(gca,'YTick',[]);

subplot(312);
pcolor(groundtruth_spec');
shading interp;
% set(gca,'XTick',[]);
% set(gca,'YTick',[]);

subplot(313);
pcolor(estimate_spec');
shading interp;
% set(gca,'XTick',[]);
% set(gca,'YTick',[]);

start = 64;
boundary = 1;
frame = 64;

newMatrix = zeros(80+boundary*2,frame*3+boundary*4);

temp = mix_spec_mel(start+1:start+frame,:);
newMatrix(boundary+1:boundary+80,boundary+1:boundary+frame) = temp';

temp = groundtruth_spec(start+1:start+frame,:);
newMatrix(boundary+1:boundary+80,boundary*2+frame+1:boundary*2+frame*2) = temp';

temp = estimate_spec(start+1:start+frame,:);
newMatrix(boundary+1:boundary+80,boundary*3+frame*2+1:boundary*3+frame*3) = temp';

figure('Position', [400, 300, 600, 250]);
% pcolor(newMatrix);
% shading interp;

imagesc(flipud(newMatrix));
set(gca,'XTick',[2, frame/2+boundary, frame+boundary+2.5, frame*1.5+boundary*2, frame*2+boundary*2+2, frame*2.5+boundary*3,  frame*3+boundary*3]);
set(gca,'XTickLabel',{'0', num2str(frame/2), '0', num2str(frame/2), '0', num2str(frame/2), num2str(frame)});
xlabel('Time frames')
ylabel('Mel-scale bins')
colorbar
rectangle('Position',[1 1 frame*3+boundary*4-1 80+boundary*2-1],'EdgeColor','k','linewidth',boundary)
rectangle('Position',[boundary+frame+1 1 frame+boundary 80+boundary*2-1],'EdgeColor','k','linewidth',boundary)

text(20,15,'Mixture','Color', 'w','FontSize',10, 'FontWeight', 'bold');
text(76,15,'Groundtruth','Color', 'w','FontSize',10, 'FontWeight', 'bold');
text(145,15,'Estimation','Color', 'w','FontSize',10, 'FontWeight', 'bold');



