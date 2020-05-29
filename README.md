# Final Project in Image Analysis and Pattern Recognition at EPFL 2020

Problem: Calculate an equation based on a video sequence. Track a lego robot travelling on a board with numbers and operators.

To run from command window: `python main.py --input src/robot_parcours_1.avi --output out/name_of_video.avi`

main flow:
 * Load video frames
 * For each frame:
    * Perform intensity normalization
    * Assume the very first frame yields a clear view of all numbers and operators, store as reference
    * Identify the __red__ arrow by region growing
    * Extract the area beneath the vehicle in this frame from the reference frame. Call this area the __candidate__
    * If there exist objects on the candidate borders: iteratively peel off border pixels until
        * _i)_ Candidate is smaller than 28x28 pixels. Render candidate invalid
        * _ii)_ Candidate is larger than 28x28 and has no objects on its borders
    * If the area looks interesting (i.e. not empty and does encapsulate its objects: send to classifier
    * Accept only the classification result if every other symbol is an operator
    * No need for further classification in the sequence if a '=' is identified. Calculate
    * Construct output frame
    
* Generate output video sequence from the output frames

__Output from test video run__

![Test drive](https://github.com/arilmad/Image-Analysis-Final-Project/blob/master/illustrations/test.gif?raw=true)


__Outputs from mock test video runs__

![Test drive](https://github.com/arilmad/Image-Analysis-Final-Project/blob/master/illustrations/mock_1.gif?raw=true)

![Test drive](https://github.com/arilmad/Image-Analysis-Final-Project/blob/master/illustrations/mock_2.gif?raw=true)

![Test drive](https://github.com/arilmad/Image-Analysis-Final-Project/blob/master/illustrations/mock_3.gif?raw=true)

