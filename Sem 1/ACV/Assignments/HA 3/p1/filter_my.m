function output = filter_my(image, filter)
% This function is intended to behave like the built in function imfilter()
% See 'help imfilter' or 'help conv2'. While terms like "filtering" and
% "convolution" might be used interchangeably, and they are indeed nearly
% the same thing, there is a difference:
% from 'help filter2'
%    2-D correlation is related to 2-D convolution by a 180 degree rotation
%    of the filter matrix.

% Your function should work for color images. Simply filter each color
% channel independently.

% Your function should work for filters of any width and height
% combination, as long as the width and height are odd (e.g. 1, 7, 9). This
% restriction makes it unambigious which pixel in the filter is the center
% pixel.

% Boundary handling can be tricky. The filter can't be centered on pixels
% at the image boundary without parts of the filter being out of bounds. If
% you look at 'help conv2' and 'help imfilter' you see that they have
% several options to deal with boundaries. You should simply recreate the
% default behavior of imfilter -- pad the input image with zeros, and
% return a filtered image which matches the input resolution. A better
% approach is to mirror the image content over the boundaries for padding.

% % Uncomment if you want to simply call imfilter so you can see the desired
% % behavior. When you write your actual solution, you can't use imfilter,
% % filter2, conv2, etc. Simply loop over all the pixels and do the actual
% % computation. It might be slow.
% output = imfilter(image, filter);


%%%%%%%%%%%%%%%%
% Your code here
%%%%%%%%%%%%%%%%

% one dimension filter
function simpleFilter = simpleFilter(image, filter)
% get height and width of images and padding filter    
    imgHeight = size(image, 1);
    imgWidth = size(image, 2);
    filterHeight = size(filter, 1);
    filterWidth = size(filter, 2);
    paddingHeight = (filterHeight - 1) / 2; %padding vertically
    paddingWidth = (filterWidth - 1) / 2;   %padding horizontally
    
    simpleFilter = zeros(imgHeight, imgWidth);
    pad_oneDimensionOut = zeros(imgHeight + 2*paddingHeight, imgWidth + 2*paddingWidth); %padding the images

% implement function:
    pad_oneDimensionOut(1 + paddingHeight: imgHeight + paddingHeight, 1 + paddingWidth: imgWidth + paddingWidth) = image;
    for i = 1: imgHeight
        for j = 1: imgWidth
            im = pad_oneDimensionOut(i: i + 2*paddingHeight, j: j + 2*paddingWidth); %get the small matrix surround this pixel
            simpleFilter(i, j) = sum(sum(im .* filter));
        end
    end    
end


output = zeros(size(image));
dimension = length(size(image));
    if (dimension == 1)    % for gray scale
        output = simpleFilter(image, filter);
    elseif (dimension == 3) %for RGB image
        for k = 1: 3
            output(:, :, k) = simpleFilter(image(:, :, k), filter);
        end
    end
end

