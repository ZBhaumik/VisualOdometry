# VisualSLAM

 TODO: A better writeup/blog post about the specifics of my code, implementation, better figures, etc. is coming soon :rocket:! I am putting some neat preliminary results here, however.

I wrote a robust monocular SLAM pipeline for use in autonomous systems, and have been benchmarking it on the KITTI driving dataset, making this suitable for an autonomous driving scenario. The pipeline uses an ORB (Oriented FAST and Rotated BRIEF) feature tracker to compute keypoints and descriptors at each image, and then the keypoints are tracked across images using a Brute-Force (distance-based) matcher. Then, I use RANSAC for (semi) robust pose estimation, allowing me to track the odometry of the vehicle across frames. This gives me results like this (~96% accurate):

![Figure_00](https://github.com/user-attachments/assets/718df6fa-7e5e-4efd-8ac3-a2822f57dfad)

This allows me to do the localization part of S(L)AM. The next part is the mapping, which I think is really neat. Using the tracked keypoints across frames, the relative poses computed by RANSAC, and the focal parameters inherent to the camera, I am able to (approximately) triangulate the 3D positions of the points across frames. This allows me to create maps of the scene, as I know the approximate (x, y, z) position of my suggested points relative to my estimated pose, and can prevent duplication of points using the descriptors mentioned earlier. This gives me maps similar to this:

<img src="https://github.com/user-attachments/assets/0b3ab348-5b60-4865-ac20-96faafb6a3f7" width="500" />
<img src="https://github.com/user-attachments/assets/ac221182-e0e8-41e8-98b7-56847ee0265c" width="500" />

(the start point is marked with a red flag, to make comparisons to the 2D map above).

Obviously, this is an imperfect map of the scene; small errors in the relative pose from frame to frame compound over time, causing the computed point cloud path to drift significantly. With all of this "estimation" and "relative computation", it makes us need some way of doing an ABSOLUTE correction over all of our data - and this is what bundle adjustment is.

Bundle adjustment is very simple at its core - it is a giant regression problem. Given parameters of the poses and 3D points, we compute what the 3D points would look like projected back onto the camera, and compare this to where we saw the points on the camera screen when they were first observed. The loss function is just the difference between these two, and enforcing consistency across frames allows us to get a much more refined version of the graph.

[FIGURE SOON!]

^ The bundle adjustment I am running works most of the way, but I am trying to get really refined results right now - I am running into issues now with points being computed as outliers when they are "reprojected" in the loss function, due to tiny "point depths" in the (x, y, w) homogenous coordinate system, so I am looking at ways to filter points and make an impressive and accurate point cloud using only monocular data. Soon, I'll add other features, such as loop closure, which will only refine the odometry further.
