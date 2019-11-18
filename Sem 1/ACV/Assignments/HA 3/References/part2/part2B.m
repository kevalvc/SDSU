
close all; % closes all figures


scale = [2 1 0.5];
sz = size(scale,2);

imag = im2single(imread('image1.jpg'));
imag = rgb2gray(imag);

actual_X = 132;
actual_Y = 80;

template = im2single(imread('template.jpg'));
template = rgb2gray(template);
[fm1 fn1] = size(template);

filter = imresize(template,0.5);
[fm fn] = size(filter);

for i = 1:sz

image1 = imresize(imag,scale(i));

[im in] = size(image1);

figure(i);
subplot(4,2,1);
imshow(image1);
subplot(4,2,2);
imshow(filter);

%% Visualize and save outputs

ZMC_var =  zmc_f(image1, filter);
[mx ind] = max(ZMC_var(:));
[y, x] = ind2sub(size(ZMC_var),ind);
ZMC_error = sqrt((actual_X - x).^2+(actual_Y - y).^2);

subplot(4,2,3);
imshow(ZMC_var,[]);

ZMC1 = (ZMC_var>0.9*max(ZMC_var(:)));

subplot(4,2,4);
imshow(ZMC1,[]);

SSD = ssd_f(image1,filter);
[mx ind] = max(ZMC_var(:));
[y, x] = ind2sub(size(ZMC_var),indS);
SSD_error = sqrt((actual_X - x).^2+(actual_Y - y).^2);

subplot(4,2,5);
imshow(SSD,[]);

SSD1 = (SSD>0.9*max(SSD(:)));

subplot(4,2,6);
imshow(SSD1,[]);

NC = normxcorr2(filter,image1);

[mx ind] = max(NC(:));
[y, xp] = ind2sub(size(NC),ind);
NC_error = sqrt(((actual_X*scale(i)) - xp).^2+((actual_Y*scale(i)) - y).^2);

subplot(4,2,7);
imshow(NC,[]);

NC_OUT = (NC>0.9*max(NC(:)));

subplot(4,2,8);
imshow(NC_OUT,[]);

fprintf('Sacling Factor - %f\n',scale(i));
fprintf('Error from ZMC - %f\n',ZMC_error);
fprintf('Error from SSD - %f\n',SSD_error);
fprintf('Error from NC  - %f\n',NC_error);

end
