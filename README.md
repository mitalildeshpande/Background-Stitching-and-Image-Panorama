# Background-Stitching-and-Image-Panorama

Image Stitching:

Two or more images may have the same background but different foreground. For example, children playing in the park. They are basically moving in a scene. If this scene is being dividen into two images randomly. We need to stitch these images into one image eliminating foreground objects that move in the scene.
Steps-
Extract set of key points for each image.
Extract features from each key point.
Match features and use matches to determine if there is overlap between given pairs of images.
Compute the homography between the overlapping pairs as needed.
Transform images and stitch them into one mosaic, eliminating foreground without cropping the image.

Image Panorama:

This is the panorama feature we have in our camera. Here I have stitched multiple images into one photo.
Steps-
For each image, extract features, match features and determine the spatial overlaps of the images. 
Perform Image transformation and stitch into one panoramic photo.
