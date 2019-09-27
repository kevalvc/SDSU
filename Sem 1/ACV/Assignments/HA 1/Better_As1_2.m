loc1 = 'E:\Data\SDSU\Sem 1\ACV\Assignments\HA 1\ha1\set1\';
loc2 = 'E:\Data\SDSU\Sem 1\ACV\Assignments\HA 1\ha1\set2\';
ds = imageDatastore(loc1);
c = 0;
sumOfImage1 = zeros(215, 300, 3, 'double');
while hasdata(ds) 
    newimg = (read(ds));        
    sumOfImage1 = sumOfImage1 + im2double(newimg);
    c = c+1;
end
sumOfImage1 = sumOfImage1/c;
imshow(sumOfImage1)

ds2 = imageDatastore(loc2);
c2 = 0;
sumOfImage2 = zeros(164, 398, 3, 'double');
while hasdata(ds2) 
    newimg = (read(ds2));   
    sumOfImage2 = sumOfImage2 + im2double(newimg);
    c2 = c2+1;
end
sumOfImage2 = sumOfImage2/c2;
imshow(sumOfImage2)

subplot(1,2,1), imshow(sumOfImage1);
subplot(1,2,2), imshow(sumOfImage2);

%------------------------------------------------------------------------

listOfFiles = dir('E:\Data\SDSU\Sem 1\ACV\Assignments\HA 1\ha1\set1\*.jpg');

setsum1 = zeros(215, 300, 3, numel(listOfFiles), 'double');

setsum1_gray = zeros(215, 300, numel(listOfFiles), 'double');

for i=1:length(listOfFiles)
	imgName = ['E:\Data\SDSU\Sem 1\ACV\Assignments\HA 1\ha1\set1\' listOfFiles(i).name];
	newim = imread(imgName);
    setsum1(:,:,:,i) = im2double(newim); % New - Store the image per channel
    setsum1_gray(:,:,i) = rgb2gray(setsum1(:,:,:,i)); % New - Grayscale convert the colour image and save it
end

% Compute the average image in grayscale and colour
% Note - I would just use mean if possible
% setsum1_gray_avg = mean(setsum1_gray, 3);
% setsum1_rgb = mean(setsum1, 4);
% ... or 
% setsum1_gray_avg = sum(setsum1_gray, 3) / numel(filelist);
% setsum1_rgb = sum(setsum1, 4) / numel(filelist);
setsum1_rgb = zeros(215, 300, 3);
setsum1_gray_avg = zeros(215, 300);
for i = 1 : numel(listOfFiles)
    setsum1_rgb = setsum1_rgb + setsum1(:,:,:,i);
    setsum1_gray_avg = setsum1_gray_avg + setsum1_gray(:,:,i);
end
setsum1_rgb = setsum1_rgb / numel(listOfFiles);
setsum1_gray_avg = setsum1_gray_avg / numel(listOfFiles);

% Now compute standard deviation for each pixel
% Note - I would use std if possible
% setsum1_stddev = std(setsum1_gray, [], 3);
setsum1_stddev = zeros(215, 300);
for i = 1 : numel(listOfFiles)
    setsum1_stddev = setsum1_stddev + (setsum1_gray(:,:,i) - setsum1_gray_avg).^2;
end
setsum1_stddev = sqrt((1 / (numel(listOfFiles) - 1)) * setsum1_stddev);