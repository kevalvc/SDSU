function output = zmc_f(image, filter)

original = image;

[m n] = size(filter);
if (mod(m,2)==0)
    filter = padarray(filter,[1 0],'post');
end

if (mod(n,2)==0)
    filter = padarray(filter,[0 1],'post');
end

mn = mean(filter(:));

filter = filter-mn;

original=padarray(original,[((size(filter,1))-1)/2,((size(filter,2))-1)/2]);

for k = 1:size(image,3)
for i = 1+(((size(filter,1))-1)/2):size(image,1)+(((size(filter,1))-1)/2)
for j = 1+(((size(filter,2))-1)/2):size(image,2)+(((size(filter,2))-1)/2)
    temp = original(i-(((size(filter,1))-1)/2):i+(((size(filter,1))-1)/2),j-(((size(filter,2))-1)/2):j+(((size(filter,2))-1)/2),k) .* filter;
    out(i-(((size(filter,1))-1)/2),j-(((size(filter,2))-1)/2),k) = sum(temp(:));
end
end
end

output = out;
