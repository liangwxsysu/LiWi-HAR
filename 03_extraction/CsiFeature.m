%% LiWi-HAR: Deep Learning based Motion Segmentation Framework for Activity Recognition using WiFi
%% IEEE Internet of Things Journal 2023
%% This extracts features from CSI amplitude.
%% Designed by Vc.Liang

clc;clear;
path = 'Data_CsiAmplitude/user1/';      matUser = 'user1';  
%path = 'Data_CsiAmplitude/user2/';      matUser = 'user2';
%path = 'Data_CsiAmplitude/user3/';      matUser = 'user3';
%path = 'Data_CsiAmplitude/user4/';      matUser = 'user4';
%path = 'Data_CsiAmplitude/user5/';      matUser = 'user5';
matDir = ['Data_CsiFeature/'];                        %设置处理后数据的存储路径
SegmentFiles = dir(fullfile(path,'*.mat'));           %读取dirMat中的数据至SegmentFiles
numberFiles = length(SegmentFiles);                   %测量数据长度，即40个样本
for whichFile =1:numberFiles                          %每人的40个样本逐一循环，一个样本代表一次实验
    fprintf('seectFile  : %s, matFileName: %s\n', num2str(whichFile), SegmentFiles(whichFile).name)
    data = importdata([path,'/',SegmentFiles(whichFile).name]);       %数据加载录入
   
    %归一化标准差STD
    D=data;
    [D1,PS] = mapminmax(D');
    D1 = D1';
    STD = std(D1);
    
    %运动周期PH
    [m,n] = size(D);
    PH=m/10;
    
    %绝对中位差MAD
    MAD=median(std(D(1:end,1)-median(D)));
    
    %四分位距IQR
    Q1 = prctile(D,25);
    Q3 = prctile(D,75);
    IQR= Q3-Q1;
   
    %信息熵HI
    HI=InformationEntropy(D,10);
   
    %信号变化速率VR
    VR=sum(diff(D))/m;
    
    %均方根
    RMS=sqrt((sum(D.^2))/m);
    
    %信号均值线性偏差
    FDM =sum( abs(diff(D)))/m-1;
  
    %保存数据
    feature(whichFile,1:8)=[STD PH MAD IQR HI VR RMS FDM];

end
    label_feature = zeros(200,9);
    for i=1:5
    label_feature(40*(i-1)+1:40*i,1) = i;
    end
    label_feature(:,2:end) = feature(:,1:end);
    name = [matUser,'_','label_feature'];
    save([matDir,name], 'label_feature')

