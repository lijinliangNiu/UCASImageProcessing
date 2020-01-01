clc;
clear;
img = imread('test.png');

% ����
F = fft2(img);
Cep = ifft2(log(abs(F)));
Cep_c = fftshift(Cep);

% ��ģ���˲���L theta
% ����2���ԳƵļ�Сֵ��
Cep_cl = Cep_c;
Cep_cl(:, 257:end) = 1;
[x(1), y(1)] = find(Cep_cl == min(min(Cep_cl)));

Cep_cr = Cep_c;
Cep_cr(:, 1:256)=1;
[x(2), y(2)] = find(Cep_cr == min(min(Cep_cr)));

L = sqrt((x(1)-x(2))^2+(y(1)-y(2))^2)/2;
theta = -ceil(atan((x(2)-x(1))/(y(2)-y(1)))*180/pi);

% ����ɢ����
PSF = fspecial('motion', L, theta);

%ά���˲� ��ȥ����
wnr = deconvwnr(img, PSF, 0.01);
wnr = edgetaper(wnr, PSF);

% ���ֵ�����
GT = rgb2gray(imread('lena512color.tiff'));
fprintf('Blured: %.2f\n',psnr(img,GT));
fprintf('Restored: %.2f\n',psnr(wnr,GT));

figure;
subplot(1,4,1);
imshow(img);
title('ԭͼ');

subplot(1,4,2);
imshow(log(abs(Cep_c)),[]);
title('����');

subplot(1,4,3);
imshow(PSF,[]);
title('����ɢ����');

subplot(1,4,4);
imshow(wnr);
title('��ԭͼ��');
