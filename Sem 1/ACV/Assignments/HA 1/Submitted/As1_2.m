%-------------------------------------------------------------------------------------------------
% To compute the average image in grayscale

sumOfImage1 = zeros(215, 300, 3, 'double');
listOfFiles = dir('E:\Data\SDSU\Sem 1\ACV\Assignments\HA 1\ha1\set1\*.jpg');
for i=1:length(listOfFiles)
	imgName = ['E:\Data\SDSU\Sem 1\ACV\Assignments\HA 1\ha1\set1\' listOfFiles(i).name];
	newim = imread(imgName);
	sumOfImage1 = sumOfImage1 + im2double(newim);
end
sumOfImage1 = sumOfImage1./length(listOfFiles);
%imshow(sumOfImage1);

sumOfImage2 = zeros(164, 398, 3, 'double');
listOfFiles2 = dir('E:\Data\SDSU\Sem 1\ACV\Assignments\HA 1\ha1\set2\*.jpg');
for i=1:length(listOfFiles2)
	imgName = ['E:\Data\SDSU\Sem 1\ACV\Assignments\HA 1\ha1\set2\' listOfFiles2(i).name];
	newim = imread(imgName);
	sumOfImage2 = sumOfImage2 + im2double(newim);
end
sumOfImage2 = sumOfImage2./length(listOfFiles2);
%imshow(sumOfImage2);

%subplot(1,2,1), imshow(rgb2gray(sumOfImage1));
%subplot(1,2,2), imshow(rgb2gray(sumOfImage2));

%-------------------------------------------------------------------------------------------------
% To compute the average image in color, by averaging per RGB channel.

rgbSumOfImage1 = sumOfImage1;
rgbSumOfImage2 = sumOfImage2;
%subplot(1,2,1), imshow(rgbSumOfImage1);
%subplot(1,2,2), imshow(rgbSumOfImage2);

%-------------------------------------------------------------------------------------------------
% To compute a matrix holding the grayscale images’ standard deviation at each pixel 

listOfFiles = dir('E:\Data\SDSU\Sem 1\ACV\Assignments\HA 1\ha1\set1\*.jpg');

sumImage1 = zeros(215, 300, 3, numel(listOfFiles), 'double');

sumImage1_gray = zeros(215, 300, numel(listOfFiles), 'double');

for i=1:length(listOfFiles)
	imgName = ['E:\Data\SDSU\Sem 1\ACV\Assignments\HA 1\ha1\set1\' listOfFiles(i).name];
	newim = imread(imgName);
    sumImage1(:,:,:,i) = im2double(newim);
    sumImage1_gray(:,:,i) = rgb2gray(sumImage1(:,:,:,i));
end

sumImage_gray_avg = mean(sumImage1_gray, 3);
sumImage_rgb = mean(sumImage1, 4);
sumImage_stddev = std(sumImage1_gray, [], 3);

listOfFiles2 = dir('E:\Data\SDSU\Sem 1\ACV\Assignments\HA 1\ha1\set2\*.jpg');

sumImage2 = zeros(164, 398, 3, numel(listOfFiles2), 'double');

sumImage2_gray = zeros(164, 398, numel(listOfFiles2), 'double');

for i=1:length(listOfFiles2)
	imgName = ['E:\Data\SDSU\Sem 1\ACV\Assignments\HA 1\ha1\set2\' listOfFiles2(i).name];
	newim = imread(imgName);
    sumImage2(:,:,:,i) = im2double(newim);
    sumImage2_gray(:,:,i) = rgb2gray(sumImage2(:,:,:,i));
end

sumImage_gray_avg_2 = mean(sumImage2_gray, 3);
sumImage_rgb_2 = mean(sumImage2, 4);
sumImage_stddev_2 = std(sumImage2_gray, [], 3);

% -------------------------------------------------------------------------------------------------------
% Displaying each of them

subplot(3,2,1), imshow(rgb2gray(sumOfImage1)), title('BnW Avg 1');
subplot(3,2,2), imshow(rgb2gray(sumOfImage2)), title('BnW Avg 2');

subplot(3,2,3), imshow(rgbSumOfImage1), title('Colored Avg 1');
subplot(3,2,4), imshow(rgbSumOfImage2); title('Colored Avg 2');

subplot(3,2,5), imshow(sumImage_stddev), title('Std-Dev 1');
subplot(3,2,6), imshow(sumImage_stddev_2), title('Std-Dev 2');

% -------------------------------------------------------------------------------------------------------
