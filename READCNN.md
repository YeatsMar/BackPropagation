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

$\color{red}{增加第三层卷积层}$

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

$\color{red}{卷积核从5*5改为3*3}$

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


## Optimization

### Data Augmentation

#### Crop

26*26 左上、左下、右上、右下

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

The above is CV result, in case of overfitting and dependency of similarity, extract 20% as test set and test set had no data augmentation.

~~~
step 0, training accuracy 0.0725436
step 25, training accuracy 0.846202
step 50, training accuracy 1
test accuracy 0.985994
~~~

#### Rotate

10 and -10 degree

~~~
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

#### Gaussian Blur

OpenCV


### Batch Normalization

input of each layer: mean, std   then input = (input-mean)/std
already implemented in TF
 

### Dense Prediction


conv(3\*3, padding=1, stride=1) - BN - ReLU - Pool -> 
conv(3\*3, padding=0, stride=2) - BN - ReLU ->
conv(3\*3, padding=1, stride=1) - BN - ReLU -> avg Pool -> BP

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

#### Rotate + Crop

20% test set (test set has no data augmentation)

~~~
step 0, training accuracy 0.0843206
step 25, training accuracy 0.984321
step 50, training accuracy 1
test accuracy 0.973389
~~~



#### Dense Prediction + Crop

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
Delete
step 0, training accuracy 0.0707975
step 25, training accuracy 0.402581
step 50, training accuracy 0.504883
step 75, training accuracy 0.560451
step 100, training accuracy 0.601372
step 125, training accuracy 0.631597
step 150, training accuracy 0.650895
step 175, training accuracy 0.664729
step 200, training accuracy 0.671937
step 225, training accuracy 0.678331
step 250, training accuracy 0.682516
step 275, training accuracy 0.684376
step 300, training accuracy 0.686933
step 325, training accuracy 0.688445
step 350, training accuracy 0.689607
step 375, training accuracy 0.689607
step 400, training accuracy 0.690653
step 425, training accuracy 0.690653
step 450, training accuracy 0.690653
step 475, training accuracy 0.690653
step 500, training accuracy 0.690653
step 525, training accuracy 0.690421
test accuracy 0.684186

step 0, training accuracy 0.0854452
step 25, training accuracy 0.452337
step 50, training accuracy 0.543827
step 75, training accuracy 0.595792
step 100, training accuracy 0.632527
step 125, training accuracy 0.655545
step 150, training accuracy 0.666473
step 175, training accuracy 0.677168
step 200, training accuracy 0.680656
step 225, training accuracy 0.685422
step 250, training accuracy 0.686933
step 275, training accuracy 0.688328
step 300, training accuracy 0.689491
step 325, training accuracy 0.689026
step 350, training accuracy 0.691002
step 375, training accuracy 0.691583
step 400, training accuracy 0.691932
step 425, training accuracy 0.690305
step 450, training accuracy 0.692513
step 475, training accuracy 0.69077
step 500, training accuracy 0.691583
step 525, training accuracy 0.692397
step 550, training accuracy 0.692513
step 575, training accuracy 0.69263
step 600, training accuracy 0.692165
step 625, training accuracy 0.692746
step 650, training accuracy 0.692746
step 675, training accuracy 0.692746
step 700, training accuracy 0.692746
step 725, training accuracy 0.692746
step 750, training accuracy 0.692746
test accuracy 0.679535

step 0, training accuracy 0.0953267
step 25, training accuracy 0.437108
step 50, training accuracy 0.536387
step 75, training accuracy 0.587654
step 100, training accuracy 0.623111
step 125, training accuracy 0.639967
step 150, training accuracy 0.653569
step 175, training accuracy 0.662404
step 200, training accuracy 0.671588
step 225, training accuracy 0.677749
step 250, training accuracy 0.680539
step 275, training accuracy 0.685073
step 300, training accuracy 0.686933
step 325, training accuracy 0.688677
step 350, training accuracy 0.689142
step 375, training accuracy 0.68984
step 400, training accuracy 0.690537
step 425, training accuracy 0.690537
step 450, training accuracy 0.690421
step 475, training accuracy 0.69077
step 500, training accuracy 0.691002
step 525, training accuracy 0.691118
step 550, training accuracy 0.690886
step 575, training accuracy 0.690886
step 600, training accuracy 0.691118
step 625, training accuracy 0.690886
step 650, training accuracy 0.691351
step 675, training accuracy 0.691118
step 700, training accuracy 0.691118
step 725, training accuracy 0.691118
step 750, training accuracy 0.691118
step 775, training accuracy 0.691118
test accuracy 0.686047

step 0, training accuracy 0.0877703
step 25, training accuracy 0.435131
step 50, training accuracy 0.5608
step 75, training accuracy 0.601372
step 100, training accuracy 0.631481
step 125, training accuracy 0.651128
step 150, training accuracy 0.666589
step 175, training accuracy 0.671821
step 200, training accuracy 0.678912
step 225, training accuracy 0.684259
step 250, training accuracy 0.686933
step 275, training accuracy 0.691118
step 300, training accuracy 0.693792
step 325, training accuracy 0.692978
step 350, training accuracy 0.694257
step 375, training accuracy 0.694955
step 400, training accuracy 0.695303
step 425, training accuracy 0.695303
step 450, training accuracy 0.695303
step 475, training accuracy 0.69542
step 500, training accuracy 0.69542
step 525, training accuracy 0.69542
step 550, training accuracy 0.69542
step 575, training accuracy 0.69542
step 600, training accuracy 0.694141
test accuracy 0.666512

step 0, training accuracy 0.0782376
step 25, training accuracy 0.442688
step 50, training accuracy 0.550802
step 75, training accuracy 0.601488
step 100, training accuracy 0.636596
step 125, training accuracy 0.652174
step 150, training accuracy 0.662637
step 175, training accuracy 0.66903
step 200, training accuracy 0.674146
step 225, training accuracy 0.677982
step 250, training accuracy 0.682283
step 275, training accuracy 0.682864
step 300, training accuracy 0.685538
step 325, training accuracy 0.686236
step 350, training accuracy 0.686817
step 375, training accuracy 0.686003
step 400, training accuracy 0.687398
step 425, training accuracy 0.687515
step 450, training accuracy 0.687631
step 475, training accuracy 0.686933
step 500, training accuracy 0.687747
step 525, training accuracy 0.687747
step 550, training accuracy 0.687747
step 575, training accuracy 0.687747
step 600, training accuracy 0.687747
step 625, training accuracy 0.687747
test accuracy 0.701395
~~~







~~~
error on training set without mini-batch:
step 0, training accuracy 0.0454799
step 100, training accuracy 0.933873
step 200, training accuracy 0.992467
step 300, training accuracy 0.999442
step 400, training accuracy 1

mini-batch: 256
step 0, training accuracy 0.046875
step 0, training accuracy 0.078125
step 0, training accuracy 0.0664062
step 0, training accuracy 0.113281
step 0, training accuracy 0.0742188
step 0, training accuracy 0.0585938
step 0, training accuracy 0.078125
step 0, training accuracy 0.0976562
step 0, training accuracy 0.113281
step 0, training accuracy 0.144531
step 0, training accuracy 0.136719
step 0, training accuracy 0.0980392
step 25, training accuracy 0.988281
step 25, training accuracy 0.984375
step 25, training accuracy 0.980469
step 25, training accuracy 0.980469
step 25, training accuracy 0.992188
step 25, training accuracy 0.96875
step 25, training accuracy 0.976562
step 25, training accuracy 0.984375
step 25, training accuracy 0.96875
step 25, training accuracy 0.984375
step 25, training accuracy 0.988281
step 25, training accuracy 1
~~~


## Conclusion
