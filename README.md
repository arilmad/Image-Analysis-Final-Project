# Final Project in Image Analysis and Pattern Recognition at EPFL 2020

Problem: Calculate an equation based on a video sequence. Track a lego robot travelling on a board with numbers and operators.

To run from command window: `python main.py --input src/robot_parcours_1.avi --output out/name_of_video.avi`

main flow:
 * Load video frames
 * For each frame:
    * Perform intensity normalization
    * Assume the very first frame yields a clear view of all numbers and operators, store as reference
    * Identify the __red__ arrow by region growing
    * Extract the area beneath the vehicle in this frame from the reference frame
    * If the area looks interesting (i.e. not empty and does encapsulate its objects (i.e. no objects on the border)): send to classifier
    * Accept only the classification result if every other symbol is an operator
    * No need for further classification in the sequence if a '=' is identified. Calculate
    * Construct output frame
    
* Generate output video sequence from the output frames

Sample output shots using handed out test video:
![Test drive](https://github.com/arilmad/Image-Analysis-Final-Project/blob/master/illustrations/mov_1.gif?raw=true)
