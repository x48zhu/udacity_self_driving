#**Finding Lane Lines on the Road** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/gray_image.jpg "Grayscale"
[image2]: ./examples/blur_gray.jpg “Blur Grayscale”
[image3]: ./examples/edges.jpg “Edges”
[image4]: ./examples/masked_edges.jpg “Masked Edges”
[image5]: ./examples/lines.jpg “Lines”
[image6]: ./examples/mark_image.jpg “Final Image with Lane Detected”

---

### Reflection
 
###1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, and get rid of spurious gradients in order to prevent noise on Canny edge detection by Gaussian blur. Then I apply the Canny edge detection on the grayscale image followed by masking out edges in uninterested area (e.g. sky, trees). Next, Hough Transform is used to Find Lines from only interesting Canny Edges. Finally I combine the detected lanes with origin image.

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by grouping single lines into two left and right lines based on a range of slopes. Then find the average slope by a linear regression on all the points for both groups. (For more details I used robust linear regression which gives less weight for outliers). Then I draw the line with the average slope and corresponding X range.

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]


###2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when there are any object on current lanes (e.g. traffic sign on the road), whose edges will be detected and also within the interesting region. This will be included in the calculation and affect the result

Another shortcoming could be most the parameters are tuned manually and hardcoded in the pipeline. If there are any change in the environment, the pipeline may not work.

###3. Suggest possible improvements to your pipeline

A possible improvement would be to mask out the central region. In other word, shrink the interesting area.

Another potential improvement could be to use more complicated cv method or deep learning to detect/decide lanes.