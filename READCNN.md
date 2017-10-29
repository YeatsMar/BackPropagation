# Convolutional Neural Network

In my opinion, a **convolutional neural network** (**CNN**, or **ConvNet**) is a variation of **BackPropagation Network** (**BPNet**). In this document, I will try to illustrate my understanding of *CNN*, including  the difference or say addition of *CNN* to *BPNet*, how *CNN* works to "understand" image, and my implementation. Also, I did some experiments to optimize my CNN and will you their results.

<!-- START doctoc -->  
<!-- END doctoc -->

## Principle

### Why CNN?

#### Efficiency

Using *BPNet* to classify images, we can easily see a **exponential burst** of *weights*. Because a N\*N \- pixel image needs N\*N inputs (if it is grayscale, it would be 3\*N\*N for an RGB image and 4\*N\*N for an RGBA image), then the weights of the first hidden layer would be N\*N\*X, X is the number of units on the first hidden layer, since BPNet is fully-connected.   

It is still alright for our lab, since our images are only 28\*28 grayscale and the dataset is rather small. However, it is not adviseable to process 1024\*968 RGB images with that, which is a more common case. 

#### Accuracy

Also, the accuracy of *BPNet* on image processing is not good enough (only around **85%** in my previous lab) since the one\-dimensional input of an image sacrifice its two\-dimensional structure. In another word, the **spacial characteristic** of an image is lost. Intuitively, it is hard for human eyes to recognize an image if the image is flattened into a line. 

The input of *CNN* still maintains the structure of a raw image including its color channel. And the *depth* of *filters* is even able to reveal more topological characteristics. Without any optimization, my first experiment can easily achieves an accuracy of over **96%** on 5\-fold cross validation dataset.

### How CNN works \- Demystify the principle of CNN

From a computer's perspective, every image is an arrangement of dots (a pixel) arranged in a special order. 

![](forREADME/pic.png)

In order to detect a specific image, a filter (also an image can be represented as a matrix, just relatively much smaller) is used to identify a characteristic of the image. By sliding the filter on the image, the characteristic will be matched through convolution \- dot multiplication and summation of matrices of  corresponding region and filter.  


![](forREADME/pic1.png)

![](forREADME/pic2.png)

When the filter is slided onto the "right" place where characteristic is matched, the value of the convolution is large. While the filter is on a region where the characteristic can be hardly matched, the value will be small and close to 0.

![](forREADME/pic3.png)

After the filter convolves the whole image, an activation image is attained. And the large-value region is just where the characteristic is matched. 

The idea is originated from biological processes in which individual cortical neurons respond to stimuli only in a restricted region of the visual field known as the receptive field. The receptive fields of different neurons partially overlap such that they cover the entire visual field.

Therefore, based on several filters identifying different characteristics, an image can be detected. 

*CNN* based on *BPNet* can automatically learn proper filters to identify patterns of specific images. Thus, usually the filters are not as human-understandable as in the above illustration. And in practice, there are at least two layers of filters. The filters on the first layer aim to identify the boundaries(lines) and then the second layer aims to identify the compositions of boundaries because a filter has depth to go through several previous layers. The more convolutional layers are, the more complicated pattern can be identified.

The output of convolutional layer contains the information that most dedicates to the classification of the image. Thus from another perspective, the convolution layer aims to  extract hidden patterns and input the extracted information to *BPNet* for classification regression. 

## Implemantation
My implementation starts from the example on *Tensorflow*. 

~~~python
def deepnn(x):
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    # First convolutional layer - maps one grayscale image to 32 feature maps, i.e. 32 filters.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32]) 
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64. A filter go through all previous 32 feature maps and there are 64 filters in total.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # TODO: ReLU

    # Dropout - controls the complexity of the model, prevents co-adaptation of features.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 14 classes, one for each Chinese character
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 14])  # TODO: output size
        b_fc2 = bias_variable([14])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob

~~~

![](forREADME/mnist_deep.png)

There are two convolutional layers with 32 and 64 filters respectively. And the second convolutional layer goes through all the 32 feature maps generated from the first layer. For the sake of efficiency, there is a pooling layer after each convolutional layer. Then the reduced input are dumped into BPNet with a dropout layer between two fully connected layers.

As proposed in the example codes, a much more complex AdamOptimizer is used instead of Gradient Descent.

This model achieves an average accuracy of **0.9695848** based on 5-fold cross validation.


N^th fold  | Accuracy of Training set | Accuracy of Test set
------------- | ------------- | -------------
1  | 1 | 0.979079
2  | 1 | 0.983264
3  | 1 | 0.95537
4  | 1 | 0.969317
5  | 1 | 0.960894


~~~
step 0, training accuracy 0.161144
step 25, training accuracy 0.983955
step 50, training accuracy 0.999651
step 75, training accuracy 1
test accuracy 0.979079

step 0, training accuracy 0.105337
step 25, training accuracy 0.985351
step 50, training accuracy 0.999651
step 75, training accuracy 1
test accuracy 0.983264

step 0, training accuracy 0.118242
step 25, training accuracy 0.988141
step 50, training accuracy 1
test accuracy 0.95537

step 0, training accuracy 0.129055
step 25, training accuracy 0.993024
step 50, training accuracy 1
test accuracy 0.969317

step 0, training accuracy 0.152371
step 25, training accuracy 0.985356
step 50, training accuracy 0.999651
step 75, training accuracy 1
test accuracy 0.960894
~~~


## Optimization

### Filter 

Nowadays, it is common to use 3\*3 filter for small images and 7\*7 for the large. Thus I tried to change the filter from 5\*5 to 3\*3. And the accuracy was improved to **0.9810274** on average based on 5-fold cross validation. And since it is an obvious and stable improvement, it was used along all other optimization methods.

N^th fold  | Accuracy of Training set | Accuracy of Test set
------------- | ------------- | -------------
1  | 1 | 0.97629
2  | 1 | 0.990237
3  | 1 | 0.967922
4  | 1 | 0.987448
5  | 1 | 0.98324


~~~
step 0, training accuracy 0.122776
step 25, training accuracy 0.981514
step 50, training accuracy 0.997558
step 75, training accuracy 1
test accuracy 0.97629

step 0, training accuracy 0.193582
step 25, training accuracy 0.976979
step 50, training accuracy 0.996861
step 75, training accuracy 0.999302
step 100, training accuracy 1
test accuracy 0.990237

step 0, training accuracy 0.138821
step 25, training accuracy 0.985002
step 50, training accuracy 0.998954
step 75, training accuracy 1
test accuracy 0.967922

step 0, training accuracy 0.208929
step 25, training accuracy 0.976979
step 50, training accuracy 0.996861
step 75, training accuracy 0.999302
step 100, training accuracy 1
test accuracy 0.987448

step 0, training accuracy 0.125523
step 25, training accuracy 0.97106
step 50, training accuracy 0.996513
step 75, training accuracy 1
test accuracy 0.98324
~~~

### Network architecture

Without any experience in optimizing *CNN*, I simply tried to add one more convolutional layer to see whether it would work. However, adding one more convolutional layer to the classical LeNet5 did not bring any good. The accuracy decreased to **0.949778** on average of 5-fold cross validation. Therefore, I had to go deeper into the design principles instead of simply doing experiments of changing number of layers.

N^th fold  | Accuracy of Training set | Accuracy of Test set
------------- | ------------- | -------------
1  | 1 | 0.934449
2  | 1 | 0.958159
3  | 1 | 0.956764
4  | 1 | 0.945607
5  | 1 | 0.953911

~~~
step 0, training accuracy 0.123823
step 25, training accuracy 0.987443
step 50, training accuracy 1
test accuracy 0.934449

step 0, training accuracy 0.0927799
step 25, training accuracy 0.980816
step 50, training accuracy 1
test accuracy 0.958159

step 0, training accuracy 0.103593
step 25, training accuracy 0.986746
step 50, training accuracy 1
test accuracy 0.956764

step 0, training accuracy 0.168818
step 25, training accuracy 0.986397
step 50, training accuracy 1
test accuracy 0.945607

step 0, training accuracy 0.150279
step 25, training accuracy 0.988842
step 50, training accuracy 1
test accuracy 0.953911
~~~


### Data Augmentation

It is universally acknowledged that large dataset can solve the issue of overfit and low accuracy with appropriate features in most cases. In our case, there are only 256 samples under each category. And the Chinese characters are relatively complex to digital numbers. Therefore, data augmentation is quite necessary. Different from the identification of animals in images, Chinese characters are of fixed structure, thus cannot be transformed. And since our samples are black and white, there is no point in changing color. Crop, slight rotate are preferred. 

#### Crop

The images are cropped into 26\*26 size from four directons and enlarged to original 28\*28 size. In this way, the information will not be missed and different samples are created. We can see it from our naked eyes. This method improved accuracy to **0.995257** on average of 5-fold cross validation.

![original](forREADME/original.png) ![original](forREADME/left-up.png) ![original](forREADME/right-up.png) ![original](forREADME/left-down.png) ![original](forREADME/right-down.png)

~~~python
def crop_image_part(im, box):
    partial = im.crop(box)
    partial = partial.resize((28, 28), Image.ANTIALIAS)
    pixels = partial.load()
    for x in range(partial.width):
        for y in range(partial.height):
            pixels[x, y] = 0 if pixels[x, y] == 255 else 1
    # partial.show()
    pixels = np.array(partial.getdata())
    pixels.shape = partial.width * partial.height
    return pixels
~~~ 


N^th fold  | Accuracy of Training set | Accuracy of Test set
------------- | ------------- | -------------
1  | 1 | 0.99442
2  | 1 | 0.995815
3  | 1 | 0.99442
4  | 1 | 0.99721
5  | 1 | 0.99442


~~~
step 0, training accuracy 0.587123
step 25, training accuracy 0.998884
step 50, training accuracy 1
test accuracy 0.99442

step 0, training accuracy 0.591378
step 25, training accuracy 0.998814
step 50, training accuracy 1
test accuracy 0.995815

step 0, training accuracy 0.579171
step 25, training accuracy 0.999093
step 50, training accuracy 1
test accuracy 0.99442

step 0, training accuracy 0.51409
step 25, training accuracy 0.996652
step 50, training accuracy 1
test accuracy 0.99721

step 0, training accuracy 0.540109
step 25, training accuracy 0.998186
step 50, training accuracy 1
test accuracy 0.99442
~~~

In case of overfit and dependency of similarity, I extracted 20% original dataset as test set and test set had no data augmentation. The accuracy is still as high as **0.985994**.

~~~
step 0, training accuracy 0.0725436
step 25, training accuracy 0.846202
step 50, training accuracy 1
test accuracy 0.985994
~~~

#### Rotate

Slightly rotate original image by 10 and -10 degree, the accuracy is also improved to **0.995536** on average of cross validation and **0.977591** on 20% test set.

![](forREADME/original.png) ![](forREADME/10.png) ![](forREADME/-10.png) 


N^th fold  | Accuracy of Training set | Accuracy of Test set
------------- | ------------- | -------------
1  | 1 | 0.99721
2  | 1 | 0.99721
3  | 1 | 0.99442
4  | 1 | 0.992746
5  | 1 | 0.996094


~~~
???
step 0, training accuracy 0.582938
step 25, training accuracy 0.998047
step 50, training accuracy 1
test accuracy 0.99721

step 0, training accuracy 0.531529
step 25, training accuracy 0.998884
step 50, training accuracy 1
test accuracy 0.99721

step 0, training accuracy 0.516253
step 25, training accuracy 0.998047
step 50, training accuracy 1
test accuracy 0.99442

step 0, training accuracy 0.581124
step 25, training accuracy 0.998744
step 50, training accuracy 1
test accuracy 0.992746

step 0, training accuracy 0.555106
step 25, training accuracy 0.997977
step 50, training accuracy 1
test accuracy 0.996094
~~~


~~~
???4832
step 0, training accuracy 0.0723577
step 25, training accuracy 0.588269
step 50, training accuracy 0.690244
step 75, training accuracy 0.674448
step 100, training accuracy 0.690476
step 125, training accuracy 0.690476
step 150, training accuracy 0.68885
step 175, training accuracy 0.686992
step 200, training accuracy 0.690476
step 225, training accuracy 0.690476
test accuracy 0.977591
~~~


### Batch Normalization

Referencing the ground-breaking research *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift* in year 2015, I tried Batch Normalization in my dataset.

> Training Deep Neural Networks is complicated by the fact that the distribution of each layer’s inputs changes during training, as the parameters of the previous layers change. This slows down the training by requiring lower learning rates and careful parameter initialization, and makes it notoriously hard to train models with saturating nonlinearities. We refer to this phenomenon as internal covariate shift, and address the problem by normalizing layer inputs. Our method draws its strength from making normalization a part of the model architecture and performing the normalization for each training mini-batch. Batch Normalization allows us to use much higher learning rates and be less careful about initialization. It also acts as a regularizer, in some cases eliminating the need for Dropout. Applied to a state-of-the-art image classification model, Batch Normalization achieves the same accuracy with 14 times fewer training steps, and beats the original model by a significant margin.


That is to say, the input of each layer will be normalized as follows:

![](forREADME/BN.png)
![](forREADME/BN_formular.png)
![](forREADME/BN1.png)

Among the formulas, $\gamma$ and $\beta$ are parameters to be learnt. The last step is necessary because normalization changes the original distribution of previou output and thus may cause information loss. When $\gamma = \sqrt{Var[x]}$, $\beta = E[x]$, the original data can be restored.



Luckily, there are implenmented interface in *TensorFlow*.

~~~python
batchSize = 256
...
bn1 = tf.layers.batch_normalization(conv1_out, scale=False, training=mode)
~~~

Whether it is training or testing matters a lot. Since in training process, $\mu$ and $\sigma$ value is re-calculated and changed, while on test set, they are fixed.
$$
E[x] = E[\mu]$$
$$
Var[x] = \frac{m}{m-1}E[\sigma^2]
$$

However, the accuracy is not so good.

~~~
step 0, training accuracy 0.0874564
step 25, training accuracy 0.625087
step 50, training accuracy 0.762369
step 75, training accuracy 0.828223
step 100, training accuracy 0.858885
step 125, training accuracy 0.897213
step 150, training accuracy 0.896167
step 175, training accuracy 0.891638
step 200, training accuracy 0.937282
step 225, training accuracy 0.916725
step 250, training accuracy 0.931707
step 275, training accuracy 0.946341
step 300, training accuracy 0.950871
step 325, training accuracy 0.958537
step 350, training accuracy 0.946341
step 375, training accuracy 0.967596
step 400, training accuracy 0.974913
step 425, training accuracy 0.974564
step 450, training accuracy 0.956794
step 475, training accuracy 0.960627
step 500, training accuracy 0.966551
step 525, training accuracy 0.973171
step 550, training accuracy 0.973519
step 575, training accuracy 0.973519
step 600, training accuracy 0.975261
step 625, training accuracy 0.977352
step 650, training accuracy 0.958537
step 675, training accuracy 0.990592
step 700, training accuracy 0.966551
step 725, training accuracy 0.9777
step 750, training accuracy 0.9777
step 775, training accuracy 0.978397
step 800, training accuracy 0.976655
step 825, training accuracy 0.974564
step 850, training accuracy 0.983972
step 875, training accuracy 0.988153
step 900, training accuracy 0.981533
step 925, training accuracy 0.990941
step 950, training accuracy 0.990592
step 975, training accuracy 0.985017
test accuracy 0.578431
~~~
 

### Dense Prediction
Referenced paper *Fully Convolutional Networks for Semantic Segmentation*, I removed the second 2\*2 max\_pooling layer which need to be reshaped. Instead, I changed the stride of the filters and add a third convolutional layer and an average pooling. 

Firstly, by changing the stride of filters on the second convolutional layer from 1 to 2, the size of images are cut into half, which is similar to the funcion of 2\*2 max pooling layer but with less information loss. Secondly, another convolutional layer aims to extract more complex and detailed patterns. The last average pooling layer is of the same size of its input, making the outputs 1*1 size which does not need reshape for connection with fully connected layers.


conv(3\*3, padding=1, stride=1) - ReLU - Pool ->   
conv(3\*3, padding=0, stride=2) - ReLU ->  
conv(3\*3, padding=1, stride=1) - ReLU -> avg Pool -> fc

~~~
========== conv1 ==========
 (?, 28, 28, 32)
========== pool1 ==========
 (?, 14, 14, 32)
========== conv2 ==========
 (?, 7, 7, 64)
========== conv3 ==========
 (?, 7, 7, 128)
========== pool2 ==========
 (?, 1, 1, 128)
~~~

The accuracy is improved to **0.9877218** on average of 5-fold cross validation. 


N^th fold  | Accuracy of Training set | Accuracy of Test set
------------- | ------------- | -------------
1  | 1 | 0.991632
2  | 1 | 0.988842
3  | 1 | 0.986053
4  | 1 | 0.988842
5  | 1 | 0.98324


~~~
step 0, training accuracy 0.0812696
step 25, training accuracy 0.377398
step 50, training accuracy 0.574817
step 75, training accuracy 0.686083
step 100, training accuracy 0.764213
step 125, training accuracy 0.802232
step 150, training accuracy 0.854901
step 175, training accuracy 0.887339
step 200, training accuracy 0.903383
step 225, training accuracy 0.91873
step 250, training accuracy 0.930241
step 275, training accuracy 0.940705
step 300, training accuracy 0.948727
step 325, training accuracy 0.951168
step 350, training accuracy 0.962679
step 375, training accuracy 0.965818
step 400, training accuracy 0.966167
step 425, training accuracy 0.97105
step 450, training accuracy 0.976979
step 475, training accuracy 0.979421
step 500, training accuracy 0.98256
step 525, training accuracy 0.982211
step 550, training accuracy 0.984304
step 575, training accuracy 0.986746
step 600, training accuracy 0.98849
step 625, training accuracy 0.990931
step 650, training accuracy 0.993722
step 675, training accuracy 0.994768
step 700, training accuracy 0.996512
step 725, training accuracy 0.996163
step 750, training accuracy 0.996512
step 775, training accuracy 0.99721
step 800, training accuracy 0.997558
step 825, training accuracy 0.998256
step 850, training accuracy 0.998954
step 875, training accuracy 0.999302
step 900, training accuracy 0.999651
step 925, training accuracy 0.998605
step 950, training accuracy 0.999651
step 975, training accuracy 0.999651
step 1000, training accuracy 1
test accuracy 0.991632

step 0, training accuracy 0.1015
step 25, training accuracy 0.416463
step 50, training accuracy 0.647018
step 75, training accuracy 0.734217
step 100, training accuracy 0.791768
step 125, training accuracy 0.82595
step 150, training accuracy 0.861528
step 175, training accuracy 0.886641
step 200, training accuracy 0.906523
step 225, training accuracy 0.925358
step 250, training accuracy 0.935821
step 275, training accuracy 0.945239
step 300, training accuracy 0.951517
step 325, training accuracy 0.95954
step 350, training accuracy 0.964772
step 375, training accuracy 0.97105
step 400, training accuracy 0.975235
step 425, training accuracy 0.976979
step 450, training accuracy 0.980119
step 475, training accuracy 0.981165
step 500, training accuracy 0.983258
step 525, training accuracy 0.984304
step 550, training accuracy 0.985351
step 575, training accuracy 0.989885
step 600, training accuracy 0.990582
step 625, training accuracy 0.991629
step 650, training accuracy 0.993722
step 675, training accuracy 0.995814
step 700, training accuracy 0.995117
step 725, training accuracy 0.99721
step 750, training accuracy 0.995814
step 775, training accuracy 0.997558
step 800, training accuracy 0.997907
step 825, training accuracy 0.998605
step 850, training accuracy 0.998256
step 875, training accuracy 0.998954
step 900, training accuracy 0.998954
step 925, training accuracy 0.999302
step 950, training accuracy 0.999302
step 975, training accuracy 0.999651
step 1000, training accuracy 0.999651
step 1025, training accuracy 0.999302
step 1050, training accuracy 1
test accuracy 0.988842

step 0, training accuracy 0.066969
step 25, training accuracy 0.421346
step 50, training accuracy 0.621207
step 75, training accuracy 0.720614
step 100, training accuracy 0.779212
step 125, training accuracy 0.819672
step 150, training accuracy 0.858737
step 175, training accuracy 0.878619
step 200, training accuracy 0.901291
step 225, training accuracy 0.919777
step 250, training accuracy 0.933031
step 275, training accuracy 0.940705
step 300, training accuracy 0.95082
step 325, training accuracy 0.957447
step 350, training accuracy 0.961632
step 375, training accuracy 0.970003
step 400, training accuracy 0.972445
step 425, training accuracy 0.975235
step 450, training accuracy 0.978026
step 475, training accuracy 0.981863
step 500, training accuracy 0.984653
step 525, training accuracy 0.984653
step 550, training accuracy 0.985351
step 575, training accuracy 0.987095
step 600, training accuracy 0.989536
step 625, training accuracy 0.989187
step 650, training accuracy 0.991629
step 675, training accuracy 0.993024
step 700, training accuracy 0.994419
step 725, training accuracy 0.995117
step 750, training accuracy 0.996163
step 775, training accuracy 0.996512
step 800, training accuracy 0.997558
step 825, training accuracy 0.997558
step 850, training accuracy 0.997907
step 875, training accuracy 0.998256
step 900, training accuracy 0.997907
step 925, training accuracy 0.999302
step 950, training accuracy 1
test accuracy 0.986053

step 0, training accuracy 0.066969
step 25, training accuracy 0.422742
step 50, training accuracy 0.632717
step 75, training accuracy 0.734217
step 100, training accuracy 0.797
step 125, training accuracy 0.851064
step 150, training accuracy 0.882456
step 175, training accuracy 0.903383
step 200, training accuracy 0.921172
step 225, training accuracy 0.938961
step 250, training accuracy 0.948029
step 275, training accuracy 0.954656
step 300, training accuracy 0.960586
step 325, training accuracy 0.964423
step 350, training accuracy 0.971399
step 375, training accuracy 0.974189
step 400, training accuracy 0.974189
step 425, training accuracy 0.980119
step 450, training accuracy 0.985351
step 475, training accuracy 0.984653
step 500, training accuracy 0.986746
step 525, training accuracy 0.988838
step 550, training accuracy 0.991629
step 575, training accuracy 0.993722
step 600, training accuracy 0.993373
step 625, training accuracy 0.995466
step 650, training accuracy 0.993024
step 675, training accuracy 0.994419
step 700, training accuracy 0.994768
step 725, training accuracy 0.996512
step 750, training accuracy 0.99721
step 775, training accuracy 0.997558
step 800, training accuracy 0.999302
step 825, training accuracy 0.998605
step 850, training accuracy 0.998605
step 875, training accuracy 0.999302
step 900, training accuracy 0.998954
step 925, training accuracy 0.999651
step 950, training accuracy 0.999302
step 975, training accuracy 0.999651
step 1000, training accuracy 0.999651
step 1025, training accuracy 1
test accuracy 0.988842

step 0, training accuracy 0.0711297
step 25, training accuracy 0.382497
step 50, training accuracy 0.619596
step 75, training accuracy 0.718619
step 100, training accuracy 0.776848
step 125, training accuracy 0.82357
step 150, training accuracy 0.859135
step 175, training accuracy 0.880056
step 200, training accuracy 0.908996
step 225, training accuracy 0.921897
step 250, training accuracy 0.937239
step 275, training accuracy 0.945955
step 300, training accuracy 0.955718
step 325, training accuracy 0.960251
step 350, training accuracy 0.965132
step 375, training accuracy 0.969665
step 400, training accuracy 0.974547
step 425, training accuracy 0.978731
step 450, training accuracy 0.980474
step 475, training accuracy 0.986402
step 500, training accuracy 0.987448
step 525, training accuracy 0.989191
step 550, training accuracy 0.990934
step 575, training accuracy 0.991283
step 600, training accuracy 0.992678
step 625, training accuracy 0.995119
step 650, training accuracy 0.994421
step 675, training accuracy 0.997211
step 700, training accuracy 0.998257
step 725, training accuracy 0.997211
step 750, training accuracy 0.997211
step 775, training accuracy 0.999303
step 800, training accuracy 0.999303
step 825, training accuracy 1
test accuracy 0.98324
~~~

### More combo

#### Batch Normalization + Crop

Incredibly low accuracy of **0.159664** on 20% test set.

~~~
step 0, training accuracy 0.0919164
step 25, training accuracy 0.369477
step 50, training accuracy 0.470453
step 75, training accuracy 0.518955
step 100, training accuracy 0.552056
step 125, training accuracy 0.628571
step 150, training accuracy 0.696516
step 175, training accuracy 0.740767
step 200, training accuracy 0.789199
step 225, training accuracy 0.812125
step 250, training accuracy 0.810871
step 275, training accuracy 0.821603
step 300, training accuracy 0.827944
step 325, training accuracy 0.857631
step 350, training accuracy 0.874286
step 375, training accuracy 0.869965
step 400, training accuracy 0.889408
step 425, training accuracy 0.9023
step 450, training accuracy 0.874843
step 475, training accuracy 0.895192
step 500, training accuracy 0.879861
step 525, training accuracy 0.901324
step 550, training accuracy 0.884808
step 575, training accuracy 0.882648
step 600, training accuracy 0.872822
step 625, training accuracy 0.909826
step 650, training accuracy 0.907178
step 675, training accuracy 0.900418
step 700, training accuracy 0.886202
step 725, training accuracy 0.881185
step 750, training accuracy 0.898118
step 775, training accuracy 0.897631
step 800, training accuracy 0.88676
step 825, training accuracy 0.893589
step 850, training accuracy 0.874355
step 875, training accuracy 0.879721
step 900, training accuracy 0.887596
step 925, training accuracy 0.871289
step 950, training accuracy 0.880697
step 975, training accuracy 0.890453
test accuracy 0.159664
~~~

#### Rotate + Crop

avg: 0.865419


N^th fold  | Accuracy of Training set | Accuracy of Test set
------------- | ------------- | -------------
1  | 0.868959 | 0.860702
2  | 0.868161 | 0.863491
3  | 0.866168 | 0.872061
4  | 0.868959 | 0.874723
5  | 0.869905 | 0.856118


~~~
step 0, training accuracy 0.547334
step 25, training accuracy 0.868411
step 50, training accuracy 0.868261
step 100, training accuracy 0.868959
step 125, training accuracy 0.868959
step 150, training accuracy 0.869008
step 175, training accuracy 0.869008
...
step 1975, training accuracy 0.868959
test accuracy 0.860702

step 0, training accuracy 0.555107
step 25, training accuracy 0.86582
step 50, training accuracy 0.867613
step 75, training accuracy 0.867962
step 100, training accuracy 0.868161
step 125, training accuracy 0.868161
...
step 1975, training accuracy 0.868161
test accuracy 0.863491

step 0, training accuracy 0.589586
step 25, training accuracy 0.864474
step 50, training accuracy 0.866168
step 75, training accuracy 0.86567
step 100, training accuracy 0.866168
step 125, training accuracy 0.866168
...
step 1975, training accuracy 0.866168
test accuracy 0.872061

step 0, training accuracy 0.573792
step 25, training accuracy 0.868161
step 50, training accuracy 0.86871
step 75, training accuracy 0.868809
step 100, training accuracy 0.868959
step 125, training accuracy 0.86861
step 150, training accuracy 0.868959
...                                            
step 1975, training accuracy 0.868959

step 0, training accuracy 0.574639
step 25, training accuracy 0.869258
step 50, training accuracy 0.870055
step 75, training accuracy 0.870055
step 100, training accuracy 0.869905
step 125, training accuracy 0.869905
step 150, training accuracy 0.870055
step 175, training accuracy 0.869905
success
test accuracy 0.856118
~~~


20% test set (test set has no data augmentation)

~~~

~~~



#### Dense Prediction + Crop

avg: 0.9970426


N^th fold  | Accuracy of Training set | Accuracy of Test set
------------- | ------------- | -------------
1  | 1 | 0.996931
2  | 1 | 0.997768
3  | 1 | 0.996931
4  | 1 | 0.999163
5  | 1 | 0.99442


~~~
step 0, training accuracy 0.115653
step 25, training accuracy 0.784528
step 50, training accuracy 0.924037
step 75, training accuracy 0.960658
step 100, training accuracy 0.973145
step 125, training accuracy 0.984235
step 150, training accuracy 0.988839
step 175, training accuracy 0.993792
step 200, training accuracy 0.996024
step 225, training accuracy 0.997349
step 250, training accuracy 0.997907
step 275, training accuracy 0.998465
step 300, training accuracy 0.997907
step 325, training accuracy 0.998814
step 350, training accuracy 0.998814
step 375, training accuracy 0.999442
step 400, training accuracy 0.999651
step 425, training accuracy 0.99993
step 450, training accuracy 0.999791
step 475, training accuracy 1
test accuracy 0.996931

step 0, training accuracy 0.163365
step 25, training accuracy 0.794503
step 50, training accuracy 0.917969
step 75, training accuracy 0.956543
step 100, training accuracy 0.971749
step 125, training accuracy 0.979422
step 150, training accuracy 0.984515
step 175, training accuracy 0.989467
step 200, training accuracy 0.991839
step 225, training accuracy 0.995675
step 250, training accuracy 0.995954
step 275, training accuracy 0.996512
step 300, training accuracy 0.998535
step 325, training accuracy 0.998465
step 350, training accuracy 0.999581
step 375, training accuracy 0.999791
step 400, training accuracy 0.99986
step 425, training accuracy 0.99986
step 450, training accuracy 0.999721
step 475, training accuracy 0.999372
step 500, training accuracy 1
test accuracy 0.997768

step 0, training accuracy 0.113002
step 25, training accuracy 0.801828
step 50, training accuracy 0.936942
step 75, training accuracy 0.966448
step 100, training accuracy 0.978655
step 125, training accuracy 0.986328
step 150, training accuracy 0.991071
step 175, training accuracy 0.99421
step 200, training accuracy 0.996582
step 225, training accuracy 0.997349
step 250, training accuracy 0.997768
step 275, training accuracy 0.998326
step 300, training accuracy 0.999093
step 325, training accuracy 0.999233
step 350, training accuracy 0.999233
step 375, training accuracy 0.99986
step 400, training accuracy 0.99993
step 425, training accuracy 0.99993
step 450, training accuracy 1
test accuracy 0.996931

step 0, training accuracy 0.153251
step 25, training accuracy 0.825544
step 50, training accuracy 0.93192
step 75, training accuracy 0.964146
step 100, training accuracy 0.976632
step 125, training accuracy 0.986258
step 150, training accuracy 0.989397
step 175, training accuracy 0.993025
step 200, training accuracy 0.996233
step 225, training accuracy 0.996512
step 250, training accuracy 0.998605
step 275, training accuracy 0.998744
step 300, training accuracy 0.999581
step 325, training accuracy 0.99986
step 350, training accuracy 0.99986
step 375, training accuracy 0.99986
step 400, training accuracy 0.99993
step 425, training accuracy 1
test accuracy 0.999163

step 0, training accuracy 0.0994001
step 25, training accuracy 0.765276
step 50, training accuracy 0.915318
step 75, training accuracy 0.958147
step 100, training accuracy 0.975586
step 125, training accuracy 0.981934
step 150, training accuracy 0.986258
step 175, training accuracy 0.99149
step 200, training accuracy 0.99149
step 225, training accuracy 0.995884
step 250, training accuracy 0.997489
step 275, training accuracy 0.998535
step 300, training accuracy 0.998814
step 325, training accuracy 0.998884
step 350, training accuracy 0.99993
step 375, training accuracy 0.998884
step 400, training accuracy 0.999233
step 425, training accuracy 0.999442
step 450, training accuracy 0.99993
step 475, training accuracy 0.999791
step 500, training accuracy 0.999721
step 525, training accuracy 1
test accuracy 0.99442
~~~



#### Dense Prediction + Rotate

~~~
step 0, training accuracy 0.0994001
step 25, training accuracy 0.765276
step 50, training accuracy 0.915318
step 75, training accuracy 0.958147
step 100, training accuracy 0.975586
step 125, training accuracy 0.981934
step 150, training accuracy 0.986258
step 175, training accuracy 0.99149
step 200, training accuracy 0.99149
step 225, training accuracy 0.995884
step 250, training accuracy 0.997489
step 275, training accuracy 0.998535
step 300, training accuracy 0.998814
step 325, training accuracy 0.998884
step 350, training accuracy 0.99993
step 375, training accuracy 0.998884
step 400, training accuracy 0.999233
step 425, training accuracy 0.999442
step 450, training accuracy 0.99993
step 475, training accuracy 0.999791
step 500, training accuracy 0.999721
step 525, training accuracy 1
test accuracy 0.99442

step 0, training accuracy 0.153251
step 25, training accuracy 0.825544
step 50, training accuracy 0.93192
step 75, training accuracy 0.964146
step 100, training accuracy 0.976632
step 125, training accuracy 0.986258
step 150, training accuracy 0.989397
step 175, training accuracy 0.993025
step 200, training accuracy 0.996233
step 225, training accuracy 0.996512
step 250, training accuracy 0.998605
step 275, training accuracy 0.998744
step 300, training accuracy 0.999581
step 325, training accuracy 0.99986
step 350, training accuracy 0.99986
step 375, training accuracy 0.99986
step 400, training accuracy 0.99993
step 425, training accuracy 1
test accuracy 0.999163


~~~

~~~
step 0, training accuracy 0.0791676
step 25, training accuracy 0.423739
step 50, training accuracy 0.538363
step 75, training accuracy 0.587189
step 100, training accuracy 0.621716
step 125, training accuracy 0.641827
step 150, training accuracy 0.658451
step 175, training accuracy 0.66903
step 200, training accuracy 0.676471
step 225, training accuracy 0.680074
step 250, training accuracy 0.683329
step 275, training accuracy 0.686585
step 300, training accuracy 0.687863
step 325, training accuracy 0.688561
step 350, training accuracy 0.68891
step 375, training accuracy 0.689607
step 400, training accuracy 0.689607
step 425, training accuracy 0.689956
step 450, training accuracy 0.689142
step 475, training accuracy 0.690886
step 500, training accuracy 0.69077
step 525, training accuracy 0.690886
step 550, training accuracy 0.69077
step 575, training accuracy 0.69077
step 600, training accuracy 0.691118
step 625, training accuracy 0.689956
step 650, training accuracy 0.69077
step 675, training accuracy 0.691118
step 700, training accuracy 0.691118
step 725, training accuracy 0.691118
test accuracy 0.686047

step 0, training accuracy 0.0764938
step 25, training accuracy 0.440479
step 50, training accuracy 0.534527
step 75, training accuracy 0.594745
step 100, training accuracy 0.626017
step 125, training accuracy 0.650198
step 150, training accuracy 0.661242
step 175, training accuracy 0.670658
step 200, training accuracy 0.674611
step 225, training accuracy 0.681237
step 250, training accuracy 0.684027
step 275, training accuracy 0.684957
step 300, training accuracy 0.687631
step 325, training accuracy 0.689142
step 350, training accuracy 0.688445
step 375, training accuracy 0.689723
step 400, training accuracy 0.690537
step 425, training accuracy 0.690305
step 450, training accuracy 0.691351
step 475, training accuracy 0.691583
step 500, training accuracy 0.691932
step 525, training accuracy 0.692397
step 550, training accuracy 0.692397
step 575, training accuracy 0.692513
step 600, training accuracy 0.68984
step 625, training accuracy 0.692513
step 650, training accuracy 0.692513
step 675, training accuracy 0.692513
step 700, training accuracy 0.692513
test accuracy 0.682326

step 0, training accuracy 0.0744013
step 25, training accuracy 0.410602
step 50, training accuracy 0.545571
step 75, training accuracy 0.604976
step 100, training accuracy 0.632527
step 125, training accuracy 0.654034
step 150, training accuracy 0.664264
step 175, training accuracy 0.675076
step 200, training accuracy 0.680539
step 225, training accuracy 0.685538
step 250, training accuracy 0.687863
step 275, training accuracy 0.69077
step 300, training accuracy 0.690421
step 325, training accuracy 0.6917
step 350, training accuracy 0.692281
step 375, training accuracy 0.690537
step 400, training accuracy 0.692513
step 425, training accuracy 0.691351
step 450, training accuracy 0.69263
step 475, training accuracy 0.692746
step 500, training accuracy 0.693327
step 525, training accuracy 0.693327
step 550, training accuracy 0.693327
step 575, training accuracy 0.693327
step 600, training accuracy 0.691002
step 625, training accuracy 0.693095
test accuracy 0.677209

step 0, training accuracy 0.083934
step 25, training accuracy 0.438386
step 50, training accuracy 0.553243
step 75, training accuracy 0.605092
step 100, training accuracy 0.641246
step 125, training accuracy 0.656708
step 150, training accuracy 0.666473
step 175, training accuracy 0.673681
step 200, training accuracy 0.675889
step 225, training accuracy 0.680772
step 250, training accuracy 0.683446
step 275, training accuracy 0.684841
step 300, training accuracy 0.686817
step 325, training accuracy 0.687631
step 350, training accuracy 0.688212
step 375, training accuracy 0.68891
step 400, training accuracy 0.688793
step 425, training accuracy 0.68891
step 450, training accuracy 0.689375
step 475, training accuracy 0.689258
step 500, training accuracy 0.689491
step 525, training accuracy 0.689375
step 550, training accuracy 0.689491
step 575, training accuracy 0.689491
step 600, training accuracy 0.689491
step 625, training accuracy 0.689491
test accuracy 0.692558

step 0, training accuracy 0.0800977
step 25, training accuracy 0.45594
step 50, training accuracy 0.550453
step 75, training accuracy 0.596722
step 100, training accuracy 0.628458
step 125, training accuracy 0.65043
step 150, training accuracy 0.663567
step 175, training accuracy 0.669495
step 200, training accuracy 0.674611
step 225, training accuracy 0.678563
step 250, training accuracy 0.682399
step 275, training accuracy 0.685771
step 300, training accuracy 0.688096
step 325, training accuracy 0.689491
step 350, training accuracy 0.690653
step 375, training accuracy 0.691235
step 400, training accuracy 0.691235
step 425, training accuracy 0.691351
step 450, training accuracy 0.691467
step 475, training accuracy 0.691583
step 500, training accuracy 0.692048
step 525, training accuracy 0.692165
step 550, training accuracy 0.692165
step 575, training accuracy 0.692397
step 600, training accuracy 0.692397
step 625, training accuracy 0.692397
step 650, training accuracy 0.692281
step 675, training accuracy 0.692281
step 700, training accuracy 0.692281
test accuracy 0.681395
~~~



## Conclusion
