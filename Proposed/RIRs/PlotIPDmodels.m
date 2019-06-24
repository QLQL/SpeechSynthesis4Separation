clear all; close all; clc
load('G57_2205_IPDParams.mat');
% save(saveName,'IPD_mean','IPD_var','Azimuth_array')
Azimuth_array = [0, 15, -15, 30, -30, 60, -60, 90, -90];
index = [1,5,6,9];

figure('Position', [600, 400, 600, 300]);
hold on;
myColors = distinguishable_colors(8);
myColors = mat2cell(myColors,ones(1,size(myColors,1)));
H = [];
Hstr = cell(1,length(index));
ii = 0;
for i = index,
    ii = ii+1;
    temp = IPD_mean(:,i);
    color = myColors{ii};
    Hstr{ii} = num2str(Azimuth_array(i));
    sprintf(['The current azimuth is ',Hstr{ii}]);
    for t = 2:length(temp),
        if abs(temp(t)-temp(t-1))<pi,
            hh = plot([t-1,t],[temp(t-1),temp(t)],'color',color,'linewidth',2);
        else
            hh = plot([t-1,t],[temp(t-1),temp(t)],'.','color',color);
        end
        if t==2,
            H = [H,hh];
        end
    end
end
plot([0 600],[pi, pi],'--','color',[0.5,0.5,0.5]);
plot([0 600],[-pi, -pi],'--','color',[0.5,0.5,0.5]);
legend(H,Hstr)
ylim([-1.2*pi 1.2*pi])
xlim([0,550])
set(gca,'ytick',[-1:0.5:1]*pi); set(gca,'yticklabel',{'-\pi','-0.5\pi','0','0.5\pi','\pi'},'fontsize',12);
ylabel('Wrapped \beta(\omega) (-\pi,\pi)','fontsize',14)
set(gca,'xtick',[1:128:513]);
set(gca,'xticklabel',{'0','2.75','5.5','8.25','11'});
xlabel('\omega (kHz)','fontsize',14)
