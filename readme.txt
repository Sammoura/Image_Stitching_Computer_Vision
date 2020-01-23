
[1] create "input" and "output" directories

[2] put the professor provided images and my_input contents in "input"

[3] run Source.cpp

[4] check output for saved images, otherwise replaced by next task 

[5] then close all images on screen, in order to move to next step.

[6] remove comment of extra_4() and run again display all previous again but with seam      reduction/removal feature


Notes:

[1] openCV_contrib is used

[2] for extra_2 I used my own 3 images "S1.jpg" (left) "S2.jpg"(middle) "S3.jpg"(right), large images with few distinct features.

[3] automated order of stitching is implemented in extra_2_auto() for 3 images only, but takes extra time.

[4] extra_4 feature is implemented by center weighing inside ImageMatcher::stitch
