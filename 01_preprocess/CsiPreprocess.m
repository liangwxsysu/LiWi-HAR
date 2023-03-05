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
matDir = ['Data_CsiAmplitude/',matUser,'/'];          %���ô�������ݵĴ洢·��
action_files = dir(fullfile(datDir,'*.csv'));         %��ȡԭʼ����Ϊ��������
for i_text = 1:length(action_files)                   %���ݲ������ݳ��ȱ������ݿ�ִ��ѭ��
    fprintf('read dat: %s -- fileName: %s\n',  num2str(i_text),action_files(i_text).name)%���ַ������   
    file_name = action_files(i_text).name;            %�ļ������ǲ������ݵ��ļ���
    data_file = strcat(datDir,file_name);                   %��Ӧ�ļ��������ݸ�ֵ�������ļ�
    data = csvread(data_file);
    %% ������
    for i = 1:52
    data(2:end,i) = sqrt(data(2:end,2*i-1).^2 + data(2:end,2*i).^2);  
    end
    amplitude = data(2:end,1:52);
    
%     %% һά���Բ�ֵ
%     interpolation = amplitude;
%     i=1;
%     while i<size(interpolation,1)
%     if interpolation(i+1,1) - interpolation(i,1) ~=1                      %����processed�е�һ�е����кż���Ƿ��п�ȱֵ
%         new_row = [floor((interpolation(i+1,1)+interpolation(i,1))/2) (interpolation(i+1,2:end)+interpolation(i,2:end))/2];                           %�����������е�ƽ��ֵ��Ϊ��������                
%         interpolation = [interpolation(1:i,:);new_row;interpolation(i+1:end,:)];   %���Ǹ�������ֵ
%         i =i+1;
%     end
%     i = i+1;
%     end
%     meanvalue1 = [interpolation(:,1) mean(interpolation(:,2:end),2)];
    
   %% ��ͨ�˲����˲�
    [a, b] = butter(5, 0.05, 'low');                  
    lowpass = filter(a, b, amplitude);

   %% ��ֵ�˲�
%   [medium,TF,L,U,C] = filloutliers(amplitude,'linear','movmedian',5);
%   meanvalue2 = [medium(:,1) mean(medium(:,2:end),2)];
    
   %% ���ɷַ���
    proportion = 0.85 ; % ���ɷֵı���
    [coeff,~,latent] = pca(lowpass(:,1:end));
   %�����ۼƹ����ʣ�ȷ��ά��
    sum_latent = cumsum(latent/sum(latent)); 
    dimension = find(sum_latent>proportion);
    dimension= dimension(1);
    %��ά
    y1 = lowpass(:,1:end)* coeff(:,1:dimension); 
    y2 = y1(:,1)*latent(1,1)
    
    data_Y{i_text} = y2;
end

%% save file to mat�����ļ�
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
