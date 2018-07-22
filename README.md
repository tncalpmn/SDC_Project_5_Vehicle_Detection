
# **Project 5 - Vehicle Detection**

The goals / steps of this project are the following:

##### This project is from the fifth and last part of Udacity's Self Driving Car Nanodegree Term 1 and the goal is to create a pipeline that detects vehicles from images by using classification methods and get as less as false negative possible.
---
### Project Folder
Short overview about which file contains what:
* Vehicle Detection.ipynb: Jupyter Notebook which contains my pipeline
* Vehicle Detection.html: Html version of notebook
* output_images: Images that will be used in this writeup
* project_video_output: video, processed by my pipeline
* test_video_output.mp4: shorter version of processed original video
* project_video.mp4: Video to be processed
* test_video.mp4: Video to be processed (shorter version)
* test_images: Some test images that I can try my pipeline on
TODO

---
### Vehicle Detection Project

The goals / steps of this project are the following:

* Before training the model, the features have to extracted from given data set.
* I have used  Histogram of Oriented Gradients (HOG) feature extraction and color based (Spatial + Histogram) feature extraction in order to train my model.
* After extracting, features are separated as train and test data sets I have used Support Vector Machine classification in order to train my model.
* Then single frame/image has to be searched to label pixels in sub-images, in order to label it as vehicle or non-vehicle.
* For this I have used  sliding window method to extract the sub-images.
* In order to be more efficient during classification, I have limited my region of interest to an area where a vehicle is expected to be seen.
* In each window the features are extracted and checked if the sub-image contains any vehicle by fitting to SVN model.
* If yes the pixel coordinates are saved
* Then heat map method is applied to image, where the values of pixels inside rectangular boundaries are incremented by one as they occur in windows.
* By defining a threshold to this value, I have reduced the number of false positives in the image.
* These "warm" pixels are simply the ones that actually contains vehicles. And  boundaries of rectangular are detected by finding the maximum coordinates of  pixel islands.
* Finally this pipeline is applied to all video frames in order to detect vehicles in consecutive frames.
---
### Aftermath - What I have used?
Here is a list of new API functions that I have been using along this project.  

* **hog(img,orientation,pixels_per_cell,cells_per_block, block_norm,transform_sqrt,visualise,feature_vector)** -> Histogram of Oriented Gradients Function from scikit. Used for extracting features from Images
* **StandardScaler()**  -> Normalises  the summed features, so they would not dominate each others
* **LinearSVC()**  -> Support Vector Machine classifier. Tries to maximise the distance between margin and data-points. .fit() .predict .score
* **np.clip** -> Given list of values, it turns the ones that are smaller than small range to small range and bigger than bigger range to bigger range.
* **label**  -> Labels the features in an array (_scipy.ndimage.measurements_)

---

[//]: # (Image References)
[image1]: ./output_images/dataEx.png
[image2]: ./output_images/HOG.png
[image3]: ./output_images/window1.png
[image4]: ./output_images/heat.png
[image5]: ./output_images/final.png
[image8]: ./output_images/window2.png
[image9]: ./output_images/window3.png
[image10]: ./output_images/window4.png
[image11]: ./output_images/window5.png
[image12]: ./output_images/window6.png

[video1]: ./project_video.mp4


### Histogram of Oriented Gradients (HOG) and Color Features

In order to achieve better classification result I have extracted features from both color (Spatial & Color Histogram) and HOG. The combination of this methods as a function can be found in block 4 through 7 in file `Vehicle Detection.ipynb`

As in all machine learning problems, I have utilised a dataset which contains `vehicle` and `non-vehicle` images in order to train my model.

Here, it is important to teach the model also what is not a vehicle, as it would then label everything as vehicle. SVM is a linear classification method therefore we need two classes.

Here is an example of one of each classes:

![alt text][image1]

* Number of Vehicle Images in dataset: **8792**
* Number of Non-Vehicle Images in dataset: **8968**

In order to get a visualization what the hog features look like  I have used `visualise` parameter of hog function, so as a second output, I have received featured image. Here is randomly selected vehicle and a non-vehicle hog image which is extracted with parameters:

* **color space** = YCrCb
* **orientations** = 10
* **pixels_per_cell** = (8, 8)
* **cells_per_block** = (2, 2)

![alt text][image2]

In addition to that I have extracted color based features with following hyper-parameters:

* **color space** = YCrCb
* **spatial_size** = (32, 32)
* **hist_bins** = 32
* **hist_range** = (0, 256)

Amount of features I have extracted from an Image was `9048`

#### 2.Final choice of HOG parameters.

I have experimented with different parameters in order to get the best result, however as the number of extracted feature increased, the slower got the classification per image.

I have found out that `YUV` and `YCrCb` had the best score in classification, but at the end I have decided to process my video with `YCrCb` as it was delivering less false positives.

I was pretty satisfied with the score (%98.3) of my classifier therefore, I have not changed much the default parameters which we have used in the class.

#### 3. Training SVM

I have used SVM for this classification problem since it is prone to identify binary classes -> 1 (vehicle) or 0 (non-vehile).

SVM is also very robust to outliers and these does not affect the classification results easily, which was also one of the reasons I have chosen it.

After I have labeled my dataset with 1 and 0, I have shuffled and split my data with `train_test_split` in to training and test datasets by a ratio of 0.2.

Before training the data, the most importing thing is to normalize the features. We do not want one feature to be more dominant than the other just because of some pixel intensity or amount. So I have used StandardScaler function for scaling the data.

Once everything is ready to be trained, I had fed SVM with training data and performed a fit with test data in order to get the accuracy of the model.

As a result I have received an accuracy of `98.3`

### Sliding Window Search

#### 1. Implementation

Now my model is ready to identify new images and I need to provide it new data.

For this I have used sliding window technique, which is actually nothing but taking sub-images from the main image and looking if the classifier returns a value of 1 (vehicle).

However in order to be efficient not all of the image has to be visited. We can always assume that there would be no cars detected in the sky or above horizon so, i took a region of interest between the middle of the image and somewhere in the lowest part of the image.

Also we can perfectly assume that the cars further than our car would be smaller that the ones that is actually closer. I have also used this information to get different sizes and widths of windows.

Here one has to be careful with the scaling and start_of_ and end_of_ROI lines, because the scale might be too big to fit between ROI.

As a result, between block 16-21 in file `Vehicle Detection.ipynb`, I have experimented with different scaling and ROI parameters manually. Here are examples from XXSmall to XLarge window examples:

![alt text][image3]

![alt text][image8]

![alt text][image9]

![alt text][image10]

![alt text][image11]

![alt text][image12]

Note that, sometimes cars from other side of the road is also recognised by the classifier. Normally training data does not contains the front pictures of vehicles, but as both faces of the car is similar, classifier returns in this case also positive.

#### 2. Optimization

Even though my classifier has a high score, the false positives were inevitable. Especially shadows on the road are easily mixed up by the classifier.

I have played a lot with ROI and scale parameters for detecting a vehicle in an image.

One important mistake I have made was the order of features I concatenate the time when I train and the time when I extract from new sub-images, which took me some time to figure out.

---

### Video Implementation

#### 1. Video Result

Here's a [link to my video result](https://github.com/tncalpmn/SDC_Project_5_Vehicle_Detection/blob/master/project_video_output.mp4) 


#### 2. Filtering False Positives

I have used heat map technique in order to filter false positives. The rectangular boxes found by different ROI and scale parameters are summed up in an image and the pixels that are inside of these rectangular are rewarded as good pixels. If a pixel is awarded more than a user defined threshold, then the pixel is marked as important, meaning that it is part of the vehicle.

Then these island pixels are captured by a rectangular, which shows finally the detected vehicle.

![alt text][image8]
![alt text][image4]

In my code I have used a threshold of `18`, however that was also not enough for error free vehicle detection in video frames.

![alt text][image5]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* As mentioned earlier, I have spent some time on figuring out what was wrong with the classifier when detecting on sub-images/window frames. As it turned out that I have done feature concatenating on training different than when i am predicting.

* I wanted to completely eliminate the false positives by applying threshold value to heatmap. However this resulted  other important pixels to get lost.

* My pipeline will likely fail when other than 'car' vehicle shows up on the read (ex: truck, ambulance..), because training images contains mostly images of personal cars.

* Another not very reliable vehicle detection parameters are  ROI and scale parameters. I would assume there should be more powerful way to detect vehicles than sliding window technique because it is not very suitable for real time scenarios.
