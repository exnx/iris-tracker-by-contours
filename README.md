# iris-detection

This is a real time iris tracker that uses dlib to find facial features, and then create a
bounding box around the eyes.  It then uses contours to separate the sclera and
iris.  Once separated, we find the centroid of the iris to estimate the eye center.

The previous version used the less accurate haar cascades to find the eye bounding
boxes.  Even with a more accurate dlib feature extraction, this model is not immune
to lighting effects, or faces with glasses, which can occasionally confuse the model.

Requirements:

imutils
opencv 3
dlib

You can pip install these, or pip install -r requirements.txt to get them all automatically.
