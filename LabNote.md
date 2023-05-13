# OnlineDigitalHuman-3D-Reconstruction

## Final Goal

Realize a quasi real-time 3D avantar reconstruction system

idea

## Todo list

- [ ] 3Dxxxxx

======================================================

## Week (2), April 11th, 2023

#### Todo List Week (2)

- [ ] Get PointClouds using the Okulo Camera

- [ ] Debug the voting process to get accurate pose estimation

### What achieved week (2)

#### 1 Pipeline

Add the filtering process described by the essay. Now the whole pipeline is complete.

#### 2 Results

Carried out tests on Biwi database and images of my self. Now faces (with relatively eular angles) can be recognized very accurately and efficiently (within around 10 seconds).

#### 3 other

### Problems left week (2)

The pointclouds from the Okulo camera are not clean enough. Considering to apply a 2-D face recognition process to filter out objects other than human faces. A brute force alternative solution is to manually take a clean image.

### Plans for week (3)

========================================================

## Week (3-End), April 20th - May 13th, 2023

#### Todo List Week (1)

- [ ] Find the accurate positions of pupils on the 2D image

- [ ] Project the accurate positions to 3D pointCloud

- [ ] Estimate the center of the eyeballs

- [ ] Visualize the gaze

### What achieved

#### 1 Pipeline

- Implement all tasks mentioned above

#### 2 Results

- The estimation based on 2D images are quite accurate

- The locations estimated on the pointCloud are not very accurate, but 
is enough for estimating the gaze

- The gaze tracking process performed well on the biwi dataset

### Problems left

- Modify the code so that it can also perform on the Okulo dataset, i.e., the pictures taken by us. To achieve this, we need to: 
1. locate the face both on the 2D image and the 3D pointCloud. 
2. In addition, the pointClouds Okulo gets don't work well with open3D module, e.g., there are points that deviate from their real positions after rendering, which leads to the difficulty of locating the face.
3. Also, the scale of the training set is 1000:1mm. To fit the pointCloud from Okulo to our model, we need to find a proper parameter to scale it.

- Modify the performance of the code
1. The process of pose estimation runs too slow. It may stem from the large size of the training set (I trained abundant meshes), or the number of clustering (Typically, we generate 30-100 votes and cluster them to find the most plausible result. Too many votes, though boosting the validity of the result, cost too much time.)
2. The code itself is terrible and is written in bad coding style. 