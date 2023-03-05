%% LiWi-HAR: Deep Learning based Motion Segmentation Framework for Activity Recognition using WiFi
%% IEEE Internet of Things Journal 2023
%% This code trains and identifies different activities
%% Designed by Vc.Liang
%% 双隐含层BP神经网络
%% 清空环境变量
clc
clear
%% 数据提取及归一化
%载入训练数据和测试数据
path = 'Data_CsiFeature/'; 
SegmentFiles = dir(fullfile(path,'*.mat'));                    %读取dirMat中的数据至SegmentFiles
numberFiles = length(SegmentFiles);                            %测量数据长度
for i =1:numberFiles
    data = importdata([path,'/',SegmentFiles(i).name]); %数据加载录入
    D(120*(i-1)+1:120*i,:) = data;
end

total_feature = D;

%从1到n间随机排序
k=rand(1,600);
[m,n]=sort(k);

Rand_feature=total_feature(n(1:600),:);

%建立训练数据库和预测数据库
input_train=Rand_feature(1:540,2:end)';
output_train=Rand_feature(1:540,1)';

input_test=Rand_feature(541:600,2:end)';
output_test=Rand_feature(541:600,1)';

%训练样本输入输出数据归一化
[inputn,inputps]=mapminmax(input_train);
[outputn,outputps]=mapminmax(output_train,0,2);

%% BP网络训练
%初始化网络结构
%newff为前馈网络创建函数，[m,n]分别代表隐含层1和隐含层2神经元个数
%激活函数：tansig是S型正切函数，purelin是线性函数，logsig对数转移函数
%训练次数epochs=300，训练目标goal=0.00004，学习率lr=0.1
net=newff(inputn,outputn,[8,16],{'tansig','tansig','purelin'});  
net.trainParam.epochs=1000; 
net.trainParam.goal=0.00001;
net.trainParam.lr=0.1;
net.trainParam.showWindow=0;

%train训练网络，traingd梯度下降训练函数，traingdx梯度下降自适应学习率训练函数
[net,outputreal,E]=train(net,inputn,outputn);

%% BP网络预测
%预测数据归一化
inputn_test=mapminmax('apply',input_test,inputps);
 
%网络测试预测输出
an=sim(net,inputn_test);
 
%网络输出反归一化
BPoutput=mapminmax('reverse',an,outputps);

%将输出结果转化成整数
outputresult=round(BPoutput);

%% 预测准确率计算
r=0;rr=0;
for i=1:60
    if output_test(i)==outputresult(i)
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
plot(outputresult,':og')
hold on
plot(output_test,'-*');
legend('预测输出','期望输出')
title('BP网络预测取整输出','fontsize',12)
ylabel('函数输出','fontsize',12)
xlabel('样本','fontsize',12)
string = {'测试集BP预测结果';['accuracy = ' num2str(rightridio*100) '%']};
title(string)



