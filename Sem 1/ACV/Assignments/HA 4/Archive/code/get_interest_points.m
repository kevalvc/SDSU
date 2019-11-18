% Local Feature Stencil Code

% Returns a set of interest points for the input image

% 'image' can be grayscale or color, your choice.
% 'feature_width', in pixels, is the local feature width. It might be
%   useful in this function in order to (a) suppress boundary interest
%   points (where a feature wouldn't fit entirely in the image, anyway)
%   or(b) scale the image filters being used. Or you can ignore it.

% 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.
% 'confidence' is an nx1 vector indicating the strength of the interest
%   point. You might use this later or not.
% 'scale' and 'orientation' are nx1 vectors indicating the scale and
%   orientation of each interest point. These are OPTIONAL. By default you
%   do not need to make scale and orientation invariant local features.
function [x, y, confidence] = get_interest_points(image, feature_width)

% Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
% You can create additional interest point detector functions (e.g. MSER)
% for extra credit.

% If you're finding spurious interest point detections near the boundaries,
% it is safe to simply suppress the gradients / corners near the edges of
% the image.

% The lecture slides and textbook are a bit vague on how to do the
% non-maximum suppression once you've thresholded the cornerness score.
% You are free to experiment. Here are some helpful functions:
%  BWLABEL and the newer BWCONNCOMP will find connected components in 
% thresholded binary image. You could, for instance, take the maximum value
% within each component.
%  COLFILT can be used to run a max() operator on each sliding window. You
% could use this to ensure that every interest point is at a local maximum
% of cornerness.

% Placeholder that you can delete. 20 random points
alpha = 0.04;
gaussianFil = fspecial('Gaussian', [25 25], 1);
[gaussx, gaussy] = imgradientxy(gaussianFil);

imgx = imfilter(image, gaussx);
imgy = imfilter(image, gaussy);

% Supress gradients near the edges
imgx([(1:feature_width) end-feature_width+(1:feature_width)],:) = 0;
imgx(:, [(1:feature_width) end-feature_width+(1:feature_width)]) = 0;
imgy([(1:feature_width) end-feature_width+(1:feature_width)],:) = 0;
imgy(:, [(1:feature_width) end-feature_width+(1:feature_width)]) = 0;

large_gaussian = fspecial('Gaussian', [25 25], 2);
ixx = imfilter(imgx.*imgx, large_gaussian);
ixy = imfilter(imgx.*imgy, large_gaussian);
iyy = imfilter(imgy.*imgy, large_gaussian);
harrisCor = ixx.*iyy - ixy.*ixy - alpha.*(ixx+iyy).*(ixx+iyy);
thresholded = harrisCor > 10*mean2(harrisCor); %Adaptive threshold

sliding = 0;

switch sliding
    case 1
        harrisCor = harrisCor.*thresholded;
        har_max = colfilt(harrisCor, [feature_width feature_width], 'sliding', @max);
        harrisCor = harrisCor.*(harrisCor == har_max);

        [y, x] = find(harrisCor > 0);
        confidence = harrisCor(harrisCor > 0);
    case 0
        components = bwconncomp(thresholded);
        width = components.ImageSize(1);
        x = zeros(components.NumObjects, 1);
        y = zeros(components.NumObjects, 1);
        confidence = zeros(components.NumObjects, 1);
        for j=1:(components.NumObjects)
            pixel_ids = components.PixelIdxList{j};
            pixel_values = harrisCor(pixel_ids);
            [maximum_value, maximum_id] = max(pixel_values);
            x(j) = floor(pixel_ids(maximum_id)/ width);
            y(j) = mod(pixel_ids(maximum_id), width);
            confidence(j) = maximum_value;
        end
end


