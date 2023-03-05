%% LiWi-HAR: Deep Learning based Motion Segmentation Framework for Activity Recognition using WiFi
%% IEEE Internet of Things Journal 2023
%% This code identifies different activities by the trained net 
%% Designed by Vc.Liang
%% 清空环境变量
clc;
clear;
%% 数据输入
%载入训练数据和测试数据
path = 'Data_CsiFeature_Trained/'; 
SegmentFiles = dir(fullfile(path,'*.mat'));         %读取dirMat中的数据至SegmentFiles
data = importdata([path,'/',SegmentFiles(1).name]); %数据加载录入
Rand_feature=data;

%建立训练数据库和预测数据库
input_train=Rand_feature(1:540,2:end)';
output_train=Rand_feature(1:540,1)';

input_test=Rand_feature(541:600,2:end)';
output_test=Rand_feature(541:600,1)';

%% BP网络预测
%预测数据归一化
load Trianed_NET
inputn_test=mapminmax('apply',input_test,inputps);
 
%网络预测输出
an=sim(net,inputn_test);
 
%网络输出反归一化
BPoutput=mapminmax('reverse',an,outputps);

%将输出结果转化成0\1\2
output2=round(BPoutput);

%% 预测准确率计算
r=0;rr=0;
for i=1:60
    if output_test(i)==output2(i)
        r=r+1;
    else
        rr=rr+1;
    end
end

rightridio=r/60;
disp('准确率')
disp(rightridio);

%% 结果分析
%BP网络预测输出
figure
plot(BPoutput,':og')
hold on
plot(output_test,'-*');
legend('预测输出','期望输出')
title('BP网络预测输出','fontsize',12)
ylabel('函数输出','fontsize',12)
xlabel('样本','fontsize',12)
string = {'测试集BP预测结果';['accuracy = ' num2str(rightridio*100) '%']};
title(string)

%BP网络预测输出
figure
plot(output2,':og')
hold on
plot(output_test,'-*');
legend('预测输出','期望输出')
title('BP网络预测取整输出','fontsize',12)
ylabel('函数输出','fontsize',12)
xlabel('样本','fontsize',12)
string = {'测试集BP预测结果';['accuracy = ' num2str(rightridio*100) '%']};
title(string)