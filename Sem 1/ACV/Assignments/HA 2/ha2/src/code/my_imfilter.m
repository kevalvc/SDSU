function op = my_imfilter(image, filter)
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

    function output = myfil(img, filter)

        im = img;
        fil = filter;
%        im = im2double(image);
%        fil = im2double(filter);
        %im = im(1:5, 1:6)
        s1 = size(im)
        s2 = size(fil)
        %padHgt = s2(1)/2
        padHgt = floor(s2(1)/2);
        padWdt = floor(s2(2)/2);
        disp("abcde");
        finalPadMat = zeros(s1(1)+2*padHgt , s1(2)+2*padWdt);
        output = zeros(size(im, 1), size(im, 2));
        %output = zeros(size(finalPadMat, 1)-padHgt, size(finalPadMat, 2)-padWdt);
        finalPadMat(1+padHgt:(padHgt+s1(1)), 1+padWdt:(padWdt+s1(2))) = im;

        %size(output)
        %size(finalPadMat)
        %imshow(finalPadMat)
        %imshow(filter)

        for i=1:s1(1)
            for j=1:s1(2)
                result = finalPadMat(i:i+(2*padHgt), j:j+(2*padWdt));
                %size(result)
                %size(fil)
                %size(result)
                totSum = sum(sum(result.*fil));
                output(i, j) = (totSum);
            end
        end
    end
 
op = zeros(size(image));

dim3 = length(size(image));
if (dim3 == 1)
    op = myfil(image, filter);
elseif (dim3 == 3)
    for z = 1:3
        disp("1, 2, 3, 4...");
        op(:, :, z) = myfil(image(:, :, z), filter);
    end
end
end 
