%% LiWi-HAR: Deep Learning based Motion Segmentation Framework for Activity Recognition using WiFi
%% IEEE Internet of Things Journal 2022
%% This extracts amplitudes and PCA from raw CSI.dat files, and save as .matfiles.
%% Designed by Vc.Liang

clc;clear;
  datDir = 'Data_RawCSIDat/user1_Liang_Weixi_dat/';    matUser = 'user1';  
% datDir = 'Data_RawCSIDat/user2_Chen_Zhuolong_dat/';  matUser = 'user2';
% datDir = 'Data_RawCSIDat/user3_Long_Xudong_dat/';    matUser = 'user3';
% datDir = 'Data_RawCSIDat/user4_Jiang_Sihan_dat/';    matUser = 'user4';
% datDir = 'Data_RawCSIDat/user5_Wang_Ruqi_dat/';      matUser = 'user5';
matDir = ['Data_CsiAmplitude/',matUser,'/'];          %设置处理后数据的存储路径
action_files = dir(fullfile(datDir,'*.csv'));         %读取原始数据为操作数据
for i_text = 1:length(action_files)                   %根据操作数据长度遍历数据库执行循环
    fprintf('read dat: %s -- fileName: %s\n',  num2str(i_text),action_files(i_text).name)%按字符串输出   
    file_name = action_files(i_text).name;            %文件名就是操纵数据的文件名
    data_file = strcat(datDir,file_name);                   %对应文件名的数据赋值给数据文件
    data = csvread(data_file);
    %% 求解振幅
    for i = 1:52
    data(2:end,i) = sqrt(data(2:end,2*i-1).^2 + data(2:end,2*i).^2);  
    end
    amplitude = data(2:end,1:52);
    
%     %% 一维线性插值
%     interpolation = amplitude;
%     i=1;
%     while i<size(interpolation,1)
%     if interpolation(i+1,1) - interpolation(i,1) ~=1                      %利用processed中第一列的序列号检测是否有空缺值
%         new_row = [floor((interpolation(i+1,1)+interpolation(i,1))/2) (interpolation(i+1,2:end)+interpolation(i,2:end))/2];                           %利用上下两行的平均值作为补充数据                
%         interpolation = [interpolation(1:i,:);new_row;interpolation(i+1:end,:)];   %覆盖更新数据值
%         i =i+1;
%     end
%     i = i+1;
%     end
%     meanvalue1 = [interpolation(:,1) mean(interpolation(:,2:end),2)];
    
   %% 低通滤波器滤波
    [a, b] = butter(5, 0.05, 'low');                  
    lowpass = filter(a, b, amplitude);

   %% 中值滤波
%   [medium,TF,L,U,C] = filloutliers(amplitude,'linear','movmedian',5);
%   meanvalue2 = [medium(:,1) mean(medium(:,2:end),2)];
    
   %% 主成分分析
    proportion = 0.85 ; % 主成分的比例
    [coeff,~,latent] = pca(lowpass(:,1:end));
   %计算累计贡献率，确认维度
    sum_latent = cumsum(latent/sum(latent)); 
    dimension = find(sum_latent>proportion);
    dimension= dimension(1);
    %降维
    y1 = lowpass(:,1:end)* coeff(:,1:dimension); 
    y2 = y1(:,1)*latent(1,1)
    
    data_Y{i_text} = y2;
end

%% save file to mat保存文件
for i_text = 1:1:length(action_files)
    fprintf('save mat : %s -- fieName: %s\n',  num2str(i_text),action_files(i_text).name)      
    file_name = action_files(i_text).name;
    fn = [file_name(1),file_name(2),file_name(3),file_name(4)];
    name = [matUser,'_',fn,'_',file_name(6),file_name(7)];
    
    processed_data = data_Y{1,i_text};
    final_data = zeros(length(processed_data(:,1)),1,1);
    for i = 1:length(processed_data(:,1))  
        final_data(i,:,1) = processed_data(i,1:1);
    end
    %save([path, name], 'data')
    save([matDir,name], 'final_data')
end
