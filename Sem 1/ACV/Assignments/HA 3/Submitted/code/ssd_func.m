function op = ssd_f(img, fil)

[x y] = size(fil);
if (mod(x,2)==0)
    fil = padarray(fil,[1 0],'post');
end

if (mod(y,2)==0)
    fil = padarray(fil,[0 1],'post');
end

original = img;
original=padarray(original,[((size(fil,1))-1)/2,((size(fil,2))-1)/2]);

for k = 1:size(img,3)
    for i = 1+(((size(fil,1))-1)/2):size(img,1)+(((size(fil,1))-1)/2)
        for j = 1+(((size(fil,2))-1)/2):size(img,2)+(((size(fil,2))-1)/2)
             temp = (fil - original(i-(((size(fil,1))-1)/2):i+(((size(fil,1))-1)/2),j-(((size(fil,2))-1)/2):j+(((size(fil,2))-1)/2),k)).^2 ;
             finalMat(i-(((size(fil,1))-1)/2),j-(((size(fil,2))-1)/2),k) = sum(temp(:));
        end
    end
end

op = finalMat;
