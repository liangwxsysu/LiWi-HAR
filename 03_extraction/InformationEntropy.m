function Hx=InformationEntropy(y,duan)
%����ԭ�ź�Ϊ�ο���ʱ������ź���
%���룺maxf:ԭ�źŵ����������������ĵ�
%y:������Ϣ�ص�����
%duan:������Ϣ�ص�����Ҫ���ֿ�Ŀ���
%Hx:y����Ϣ��
%�����а�duan���ȷ֣����duan=10,�ͽ����з�Ϊ10�ȷ�

%A=importdata('control15.txt');
%a=A(:,12)';
%y= medfilt1(a);
% a1=A(:,3)';
% A3= medfilt1(a1);
% a2=A(:,8)';
% A4= medfilt1(a2);
% a3=A(:,9)';
% A5= medfilt1(a3);
% a4=A(:,12)';
% A12= medfilt1(a4);
% y=cat(1,A2,A3,A4,A5,A12);

x_min=min(y);
x_max=max(y);
maxf(1)=abs(x_max-x_min);
maxf(2)=x_min;
duan_t=1.0/duan;
jiange=maxf(1)*duan_t;
% for i=1:10
%  pnum(i)=length(find((y_p>=(i-1)*jiange)&(y_p<i*jiange)));
%  end
 pnum(1)=length(find(y<maxf(2)+jiange));
for i=2:duan-1
    pnum(i)=length(find((y>=maxf(2)+(i-1)*jiange)&(y<maxf(2)+i*jiange)));
end
pnum(duan)=length(find(y>=maxf(2)+(duan-1)*jiange));
%sum(pnum)
ppnum=pnum/sum(pnum);%ÿ�γ��ֵĸ���
%sum(ppnum)
Hx=0;
for i=1:duan
    if ppnum(i)==0
        Hi=0;
    else
        Hi=-ppnum(i)*log2(ppnum(i));
    end
    Hx=Hx+Hi;
end
end