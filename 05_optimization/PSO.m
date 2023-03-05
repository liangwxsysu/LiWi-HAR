%% LiWi-HAR: Deep Learning based Motion Segmentation Framework for Activity Recognition using WiFi
%% IEEE Internet of Things Journal 2023
%% BP neural network prediction based on PSO
%% Designed by Vc.Liang

%% 基于PSO和BP网络的预测
%清空环境
clc
clear

%% 读取数据
%将数据分为分类数据和标签数据
load Rand_feature
input=Rand_feature(1:600,2:9);
output=Rand_feature(1:600,1); 

%从1到n间随机排序
k=rand(1,600);
[m,n]=sort(k);

%随机抽取组成训练数据和测试数据
input_train=input(n(1:540),:)';
output_train=output(n(1:540),:)';
input_test=input(n(541:600),:)';
output_test=output(n(541:600),:)';

%数据样本输入输出数据归一化
[inputn,inputps]=mapminmax(input_train);
[outputn,outputps]=mapminmax(output_train);

%% 构建网络
%网络节点数
inputnum=8;
hiddennum=16;
outputnum=1;
D=inputnum*hiddennum+hiddennum*outputnum+hiddennum+outputnum;

%构建网络
net=newff(inputn,outputn,hiddennum);

%% 粒子群优化算法
%参数初始化
c1 = 1.49445;  %学习因子1
c2 = 1.49445;  %学习因子2

maxgen=100;    %迭代次数  
sizepop=50;    %粒子数量

Vmax=0.2;        %粒子最大移动速度
Vmin=-0.2;
popmax=1;      
popmin=-1;

%随机初始化所有粒子的位子和速度，并计算粒子的适应度
for i=1:sizepop
    pop(i,:)=rands(1,D); %初始位置
    V(i,:)=rands(1,D);     %初始速度
    fitness(i)=fun(pop(i,:),inputnum,hiddennum,outputnum,net,inputn,outputn); %初始适应度
end

% 初始个体极值和群体极值
[bestfitness,bestindex]=min(fitness);
gbest=pop;                  %个体最佳
fitnessgbest=fitness;       %个体最佳适应度值
zbest=pop(bestindex,:);     %全局最佳
fitnesszbest=bestfitness;   %全局最佳适应度值


%% 迭代寻优
for i=1:maxgen
    i;
    for j=1:sizepop
        
        %速度更新
        V(j,:) = V(j,:) + c1*rand*(gbest(j,:)-pop(j,:)) + c2*rand*(zbest-pop(j,:));
        V(j,V(j,:)>Vmax)=Vmax;
        V(j,V(j,:)<Vmin)=Vmin;
        
        %种群更新
        pop(j,:)=pop(j,:)+0.2*V(j,:);
        pop(j,pop(j,:)>popmax)=popmax;
        pop(j,pop(j,:)<popmin)=popmin;
        
        %自适应变异
        pos=unidrnd(21);
        if rand>0.95
            pop(j,pos)=5*rands(1,1);
        end
      
        %适应度值更新
        fitness(j)=fun(pop(j,:),inputnum,hiddennum,outputnum,net,inputn,outputn);
    end
    
    for j=1:sizepop
    %个体最优更新
    if fitness(j) < fitnessgbest(j)
        gbest(j,:) = pop(j,:);
        fitnessgbest(j) = fitness(j);
    end
    
    %群体最优更新 
    if fitness(j) < fitnesszbest
        zbest = pop(j,:);
        fitnesszbest = fitness(j);
    end
    
    end
    
    yy(i)=fitnesszbest;    
        
end

%% 结果分析
plot(yy,'r','Linewidth',1.5)
xlabel('Evolutionary Algebra','FontSize',12);
ylabel('Fitness','FontSize',12);
set(gca,'FontSize',11,'FontWeight','bold');

%% 把最优初始阀值权值赋予网络预测
x=zbest;
w1=x(1:inputnum*hiddennum);
B1=x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);
w2=x(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);
B2=x(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);

net.iw{1,1}=reshape(w1,hiddennum,inputnum);
net.lw{2,1}=reshape(w2,outputnum,hiddennum);
net.b{1}=reshape(B1,hiddennum,1);
net.b{2}=B2;

%% BP网络训练
%网络进化参数
net.trainParam.epochs=100;
net.trainParam.lr=0.1;
%net.trainParam.goal=0.00001;

%网络训练
[net,per2]=train(net,inputn,outputn);

%% BP网络预测
%数据归一化
inputn_test=mapminmax('apply',input_test,inputps);
an=sim(net,inputn_test);
test_simu=mapminmax('reverse',an,outputps);
error=test_simu-output_test;
