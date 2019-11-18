
close all;


imgScaling = [2 1 0.5];
imSize = size(imgScaling,2);

p = [];

img = im2single(imread('E:\Data\SDSU\Sem 1\ACV\Assignments\HA 3\Submitted\data\elonmusk1.jpg'));
img = rgb2gray(img);

template = im2single(imread('E:\Data\SDSU\Sem 1\ACV\Assignments\HA 3\Submitted\data\tempMusk.jpg'));
template = rgb2gray(template);
[t1 t2] = size(template);

filter = imresize(template,0.5);
[f1 f2] = size(filter);

ogg_X = 130;
ogg_Y = 282;

for i = 1:imSize

    imageR = imresize(img,imgScaling(i));

    [im1 im2] = size(imageR);

    figure(i);
    subplot(4,2,1);
    imshow(imageR);
    subplot(4,2,2);
    imshow(filter);

    ZMC_varr =  zmc_func(imageR, filter);
    [maxX indices] = max(ZMC_varr(:));
    [x, y] = ind2sub(size(ZMC_varr),indices);
    Zmc_err = sqrt((ogg_X - y).^2+(ogg_Y - x).^2);

    subplot(4,2,3);
    imshow(ZMC_varr,[]);

    ZMC = (ZMC_varr>0.9*max(ZMC_varr(:)));

    subplot(4,2,4);
    imshow(ZMC,[]);

    SSD_varr = ssd_func(imageR,filter);
    [maxX indices] = max(ZMC_varr(:));
    [x, y] = ind2sub(size(ZMC_varr),indices);
    Ssd_err = sqrt((ogg_X - y).^2+(ogg_Y - x).^2);

    subplot(4,2,5);
    imshow(SSD_varr,[]);

    SSD = (SSD_varr>0.9*max(SSD_varr(:)));

    subplot(4,2,6);
    imshow(SSD,[]);

    NCC_varr = normxcorr2(filter,imageR);

    [maxX indices] = max(NCC_varr(:));
    [x, xp] = ind2sub(size(NCC_varr),indices);
    Ncc_err = sqrt(((ogg_X*imgScaling(i)) - xp).^2+((ogg_Y*imgScaling(i)) - x).^2);

    subplot(4,2,7);
    imshow(NCC_varr,[]);

    NCC = (NCC_varr>0.9*max(NCC_varr(:)));

    subplot(4,2,8);
    imshow(NCC,[]);
    
    p = [p, Zmc_err];
    p = [p, Ssd_err];
    p = [p, Ncc_err];

    fprintf('Scaling Factor is:  %f\n',imgScaling(i));
    fprintf('Error in ZMC is: %f\n',Zmc_err);
    fprintf('Error in SSD is: %f\n',Ssd_err);
    fprintf('Error in NCC is: %f\n',Ncc_err);

end

%h = hist(p);
%plot(h)
