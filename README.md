# A PyTorch Implementation of Image Style Transfer Using Convolutional Neural Networks


This is a PyTorch implementation of [Image Style Transfer Using Convolutional Neural Networks](http://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html), inspired by [authors of paper repo](https://github.com/leongatys/PytorchNeuralStyleTransfer).


Coarse-to-fine high-resolution is also added, from paper [Controlling Perceptual Factors in Neural Style Transfer.](https://arxiv.org/abs/1611.07865)


### Usage
Simply run `python3 main.py`. 

You may want to change `DATA.STYLE_IMG_PATH`, `DATA.CONTENT_IMG_PATH` in config file, for transferring 
style of your desired style image to your content image.



### Example

We used the same images, as the authors.

Content Image:
![Content](./images/contents/Tuebingen_Neckarfront.jpg)

Style Image:
![Style](./images/styles/vangogh_starry_night.jpg)

Result of transferring the style to content image:
![Results](./examples/res.jpg)

Result of high resolution image:
![Results](./examples/hr_res.jpg)
