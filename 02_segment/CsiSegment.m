%% LiWi-HAR: Deep Learning based Motion Segmentation Framework for Activity Recognition using WiFi
%% IEEE Internet of Things Journal 2023
%% This code can split contiguous data.
%% Designed by Vc.Liang

clc;clear;
path = 'Data_Continuous/user1/';    matUser = 'user1';  
% path = 'Data_Continuous/user2';       matUser = 'user2';
% path = 'Data_Continuous/user3/';      matUser = 'user3';
% path = 'Data_Continuous/user4/';      matUser = 'user4';
% path = 'Data_Continuous/user5/';      matUser = 'user5';
matDir = ['Data_Segment/',matUser,'/'];               %设置处理后数据的存储路径
SegmentFiles = dir(fullfile(path,'*.mat'));           %读取dirMat中的数据至SegmentFiles
numberFiles = length(SegmentFiles);                   %测量数据长度
for whichFile =1:numberFiles                          %每个样本逐一循环，一个样本代表一次实验
    fprintf('seectFile  : %s, matFileName: %s\n', num2str(whichFile), SegmentFiles(whichFile).name)
    raw = importdata([path,'/',SegmentFiles(whichFile).name]);       %数据加载录入

for i = 1:192
    raw(:,i+1) = sqrt(raw(:,2*i).^2 + raw(:,2*i+1).^2);
end
raw1 = raw(:,1:193);

 [raw1,TF,L,U,C] = filloutliers(raw1,'linear','movmedian',5);


processed = [];
i=1;
while i<size(raw1,1)
    if raw1(i+1,1) - raw1(i,1) ~=1
        start_time = raw1(i,1);
        end_time = raw1(i+1,1)-1;
        for j=start_time:end_time
            new_row = [j raw1(i,2:end)+(j-start_time)*(raw1(i+1,2:end)-raw1(i,2:end))/(end_time-start_time+1)];
            processed = [processed;new_row];
        end
    else
        processed = [processed;raw1(i,:)];
    end
    i = i+1;
end



% PCA
[coeff,score,latent,tsquared] = pca(processed(:,2:end));
explained = 100*latent/sum(latent);%计算贡献率
cum = cumsum(explained);
processed(:,2:end)= bsxfun(@minus,processed(:,2:end),mean(processed(:,2:end),1));
i = 1;
while 1
    if(cum(i) > 80)
        break;
    end
    i = i+1;
end
y1 = processed(:,2:end)*coeff(:,1);

plot((1:length(y1))/10,y1,'Linewidth',1);xlabel('Time/s','FontSize',12);ylabel('Amplitude/dB','FontSize',12);set(gca,'FontSize',11,'FontWeight','bold');


% AACA
win = 20;
offset = 1;
y_var = zeros(length(y1)-win,1);
for i=1:length(y1)-win
    y_var(i) = 1/win*sum((y1(i*offset:i*offset+win)-mean(y1(i*offset:i*offset+win))).^2);
end



y_sum = zeros(length(y_var)-win,1);
for i=1:length(y_var)-win
    y_sum(i) = sum(y_var(i*offset:i*offset+win));
end
y_diff = y_sum(2:end)-y_sum(1:end-1);
tem3 = medfilt1(y_diff,3);

tem3 = abs(tem3);
x = 1/10*(1:length(tem3));

start_point = [];
end_point = [];
th = 300;
seq = [];
i=2;

while i<=length(tem3)
    if (tem3(i-1)-th) * (tem3(i)-th) < 0
        seq = [seq i];
    end
    i=i+1;
end
clf = zeros(4,100);
clf(1,1) = seq(1);
j = 1;
k = 2;
for i=2:length(seq)
    if seq(i) - seq(i-1) <= 110
        clf(j,k) = seq(i);
        k = k+1;
    else
        j=j+1;
        k = 1;
        clf(j,k) = seq(i);
        k = k+1;
    end
end
for i=1:4
    start_point = [start_point min(nonzeros(clf(i,:)))];
    end_point = [end_point max(nonzeros(clf(i,:)))];
end

start_point = start_point + 30;
end_point = end_point + 30;


hold on

for i=1:4
    hold on
    plot([start_point(i) start_point(i)]/10,[-50 150],'r','Linewidth',1.5);
end
for i=1:3
    hold on
    plot([end_point(i) end_point(i)]/10,[-50 150],'g','Linewidth',1.5);
end

offset2 = 20;
A1 = y1(1:start_point(1)+offset2,:);
A2 = y1(start_point(1)+offset2:end_point(1)+offset2,:);
A3 = y1(end_point(1)+offset2:start_point(2)+offset2,:);
A4 = y1(start_point(2)+offset2:end_point(2)+offset2,:);
A5 = y1(end_point(2)+offset2:start_point(3)+offset2,:);
A6 = y1(start_point(3)+offset2:end_point(3)+offset2,:);
A7 = y1(end_point(3)+offset2:start_point(4)+offset2,:);
A8 = y1(start_point(4)+offset2:end,:);

A1=[ A1; zeros(700-length(A1),1)];
A2=[ A2; zeros(700-length(A2),1)];
A3=[ A3; zeros(700-length(A3),1)];
A4=[ A4; zeros(700-length(A4),1)];
A5=[ A5; zeros(700-length(A5),1)];
A6=[ A6; zeros(700-length(A6),1)];
A7=[ A7; zeros(700-length(A7),1)];
A8=[ A8; zeros(700-length(A8),1)];
A =[A1,A2,A3,A4,A5,A6,A7,A8];
C =mat2cell(A,(700),[1,1,1,1,1,1,1,1]);

%% save file to mat保存文件
for i = 1:8     
    N=['1','j','2','w','3','r','4','f'];
    file_name = SegmentFiles(whichFile).name;            %文件名就是操纵数据的文件名
    name = [matUser,'_',N(i),'_',file_name(12),file_name(13)];
    seg_activties(:,1) = A(:,i);
    save([matDir,name], 'seg_activties');
end
    
end





