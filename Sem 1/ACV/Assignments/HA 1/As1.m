I = imread('tiger.jpg');
%imagesc(I)                    % Display it as gray level image
%colormap gray;

grayImage = rgb2gray(I);
%hist = imhist(I, 32);
mat = grayImage(1:100, 1:100);

% Q3 - a)
x = mat(:);
sizex = size(x);
sortedV = sort(x);
plot(sortedV, '-');

% Q3 - b)
histogram(mat, 32);

% Q3 - c)
%t = input("Insert a value that marks the threshold t");
t = 0.5;
binar = imbinarize(mat, t);
imshow(binar)

% Q3 - d)
meanint = mean2(grayImage(:, :));
sub = imsubtract(mat, meanint);
%imshow(sub)

% Q3 - e)
%rollDice();

% Q3 - f)
y = [1:6];
res1 = reshape(y, [3,2]);

% Q3 - g)
sing = mat(:);
x = min(sing);
[r, c] = find(mat == x);

% Q3 - h)
v = [1 8 8 2 1 3 9 8];
uni = unique(v);
nouniq = length(uni);

% Function for Q5 - e)
function dice = rollDice()
min = 1;
max = 6;
r = round(min + (max-min).*rand(1));
%r = rand();
if r == 1
    dice = 1;
elseif mod(r, 2) == 0
    dice = 2;
elseif mod(r, 3) == 0
    dice = 3;
elseif mod(r,4) == 0
    dice = 4;
elseif mod(r,5) == 0
    dice  = 5;
else
    dice = 6;
end
disp(dice);
end