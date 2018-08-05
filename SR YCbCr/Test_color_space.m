im = im2single(imread('cat_sleep_pc.jpg'));
max(im(:))
min(im(:))
figure; imshow(im)

im_YCBCR = rgb2ycbcr(im);
max(im_YCBCR(:))
min(im_YCBCR(:))

figure; imshow(cat(2, im_YCBCR(:,:,1), im_YCBCR(:,:,2), im_YCBCR(:,:,3) ))

im_RGB = ycbcr2rgb(im_YCBCR)

err = sum(reshape(abs(im_RGB- im), [],1));