im = im2single(imread('/home/z/PycharmProjects/SR/full_data_512/platonic/input_Cam044.png'));
img_ycbcr = rgb2lab(im);
img_rgb = lab2rgb(img_ycbcr);
err=sum(sum(sum(abs(img_rgb-im))));
figure,imshow(im);
figure,imshow(img_rgb);
a = min(min(min(img_ycbcr(:,:,1))));
b = min(min(min(img_ycbcr(:,:,2))));
c = min(min(min(img_ycbcr(:,:,3))));

% max(im(:))
% min(im(:))
% figure; imshow(im)
% 
% im_YCBCR = rgb2ycbcr(im);
% max(im_YCBCR(:))
% min(im_YCBCR(:))
% 
% figure; imshow(cat(2, im_YCBCR(:,:,1), im_YCBCR(:,:,2), im_YCBCR(:,:,3) ))
% 
% im_RGB = ycbcr2rgb(im_YCBCR)
% 
% err = sum(reshape(abs(im_RGB- im), [],1));