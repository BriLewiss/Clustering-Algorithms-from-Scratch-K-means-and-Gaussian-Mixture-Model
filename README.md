# Clustering Algorithms from Scratch: K-means and Gaussian Mixture Model

## 1) Abstract
In this assignment, students were to:


a)	Apply principal component analysis technique to reduce the 64 dimensions of the Digits dataset to 2 PCA components


b)	Apply K-means clustering algorithms to cluster the reduced dimension Digits dataset and the Make Blobs


c)	Apply Gaussian Mixture model (GMM) clustering algorithms to cluster the reduced dimension Digits dataset and the Make Blobs


d)	Optimize the number of PCA components and apply better initialization techniques to achieve better clustering


e)	Compare the performance between the two datasets


All steps are to be done from scratch without the usage of any machine learning library or package.

## 2) Principal Component Analysis
### 2.1) Introduction
Principal Component Analysis (PCA) is a dimension reduction technique that reduces the number of dimensions in a dataset, while still retaining the most important information. Datasets with a high degree of dimensionality can fall victim to the curse of dimensionality, causing the data to become very sparse, and lead to the model learning noise rather than real patterns. 
The steps I followed to complete PCA include the following:
1)	Calculate the mean of the X values in the Digits dataset
2)	Subtract the mean from each value
3)	Calculate the covariance matrix
4)	Calculate the eigenvalues and eigenvectors
5)	Choose components and form a feature vector
6)	Choose the top 2 eigenvectors
7)	Project the original data into the new feature space


The first two steps serve the purpose of centering the data. Centering the data allows the PCA directions to describe the variation around the average data point, centering around zero. Next, calculating the covariance matrix allows the algorithm to understand how features relate to each other,  seeing if they increase or decrease together. Defining the eigenvectors of the covariance matrix defines the directions of where the data spreads out the most, or the direction of maximum variance. The importance of these directions is defined by the eigenvalues, ranking each direction.
After calculating and ranking the eigenvalues and eigenvectors of the covariance matrix, PCA goes on to select the top k components, being two in the case of this assignment, that capture the most of the variance. The dataset is then transformed by projecting the original data onto the new feature space, or the top components.
### 2.2) Digits Dataset
In the digits dataset, each image is represented as 8 x 8 pixels, equaling 64 features each.

<img width="640" height="480" alt="digits" src="https://github.com/user-attachments/assets/39879ca2-7604-45a6-85fe-c159feda250d" />


_Figure 1. The first 5 digits of the digits dataset_

PCA looks at all 64 features simultaneously and finds directions that capture the greatest variance. 

<img width="640" height="480" alt="PCA Projection of the Digits Data" src="https://github.com/user-attachments/assets/cdf61ffe-e244-4386-a83d-fedb97882f6c" />


_Figure 2. Data plotted after PCA projection_

### 2.3) Optimization
Principal Component Analysis (PCA) can be optimized by using Proportion of Variance (PoV). PoV helps us understand how well a specific number of components explains a dataset. It allows PCA to focus on the most important features, improving performance. 
Variance measures how spread out data points are from an average. With PCA, PoV “explains how much of the total variance in the data is captured by each principal component” [8].
After using PoV to optimize the Digits dataset, it came to the conclusion that 21 components were optimal to capture and explain 90% of the variance between the data. 

<img width="640" height="480" alt="pca cumulative explained variance" src="https://github.com/user-attachments/assets/8a4e2bf5-5819-4d3c-a001-7e02d8e23ba7" />


_Figure 3. Represents the fraction of the original dataset's total variance captured by the number of principal components_

## 3) K-means Clustering Algorithm
### 3.1) Introduction
K-means clustering is an unsupervised machine learning algorithm that separates and clusters a dataset into groups based on the similarity of the points. The algorithm centers around the variable “k”, which is how many groups the data will be clustered into. The algorithm works in the following steps:
1)	K centroids are initialized 
2)	The Euclidean distance is calculated between each point to each centroid
3)	Clusters are formed, sending each data point to their nearest centroid
4)	Centroids are recalculated by finding the average of the points within each cluster
5)	Steps 1-4 repeat until either the centroid stops changing, or the max number of iterations has been reached

### 3.2) Tables and Figures
<img width="2000" height="1414" alt="k-means clustering on digits dataset" src="https://github.com/user-attachments/assets/b7d45bd5-d86e-46c9-b61d-8db924baa488" />


_Figure 4. K-means clustering on the Digits dataset with k equal to 2 through 10_

<img width="2000" height="1414" alt="blob K means" src="https://github.com/user-attachments/assets/2fabb61b-cb2d-4bbd-9729-f7028cb667b1" />


_Figure 5. K-means clustering on the Make Blob dataset with k equal to 2 through 10_

<img width="640" height="480" alt="silhouette digits" src="https://github.com/user-attachments/assets/e07ac36b-c255-45e4-b991-cfb28fa1da61" />


_Figure 6. Silhouette Score of the Digits dataset using K-means clustering_

<img width="640" height="480" alt="blob k-silhouette score" src="https://github.com/user-attachments/assets/fad7b87c-198c-456e-a32e-72a711102839" />


_Figure 7. Silhouette Score of the Make Blob dataset using K-means clustering_

### 3.3) Results Analysis
The results are analyzed through the usage of cluster scatter charts and silhouette score plotting. Silhouette score shows how similar a data point is to its own cluster compared to other groups. Ideally, intra-group similarity is high and inter-group similarity is low. The closer to +1 the score is, the better. This signifies that groups are divided into neat and strong clusters. 


On the Make Blob dataset, we can see the plot has initially well-defined clusters in four groups. The clusters are best defined at k=4, which holds true when looking at the silhouette score. Having reached a silhouette score of .80 at k=4 means that the clusters are neatly defined, with points being very close to their own group and far away from other groups.


With the Digits dataset, we can see through the clustering and the silhouette score that compressing the original 64 dimensions into just 2 dimensions did not form clearly separated groups. The silhouette score reaches a peak of .42 at k=4, showing that the groups are likely overlapping or touching.
The k-means clustering algorithm was able to perform better on the Make Blob dataset. 

## 4) Gaussian Mixture Model Clustering Algorithm
### 4.1) Introduction
Gaussian Mixture Model (GMM) clustering is an unsupervised learning technique that represents data as a mixture of Gaussian distributions. GMM can be used to form clusters within a dataset by assigning a probability that a point belongs to each cluster, rather than assigning it exclusively to one cluster. The goal is to find the parameters, being means, covariances, and mixing weights/coefficients that will best explain the observed data. 
•	**Means:** Represents the center or average value of the cluster
•	**Covariance:** Describes the variance and relationships between features, determining how spread out the data is, and the shape of each Gaussian component
•	**Mixing Weight:** Represents the prior probability or relative size/proportion of each Gaussian component


GMM uses the Expectation-Maximization (EM) algorithm to train the model. With the E-step, the probability, or responsibility, of each cluster is calculated for every data point. The M-step updates the parameters using the calculated responsibilities from the E-step.

### 4.2) Tables and Figures
<img width="2000" height="1414" alt="gmm clustering on digits dataset" src="https://github.com/user-attachments/assets/09ec227f-0fde-4256-bf2c-464f272b3a22" />


_Figure 8. Gaussian Mixture Model Clustering on the Digits dataset with number of clusters equal to 2 through 10_

<img width="2000" height="1414" alt="blob gmm" src="https://github.com/user-attachments/assets/73489531-fc58-45f0-9dcc-ebb7b190e55d" />


_Figure 9. Gaussian Mixture Model Clustering on the Make Blob dataset with number of clusters equal to 2 through 10_

<img width="640" height="480" alt="silhouette score gmm" src="https://github.com/user-attachments/assets/5c1879eb-b7ae-4fde-a73f-1683f86a9bfa" />


_Figure 10. Gaussian Mixture Model clustering Silhouette Score on the Digits dataset_

<img width="640" height="480" alt="blob gmm silhouette score" src="https://github.com/user-attachments/assets/6d0ef794-2630-4ed1-9160-45325125ed04" />


_Figure 11. Gaussian Mixture Model clustering Silhouette Score on the Make Blob dataset_

### 4.3) Results Analysis
The results for Gaussian Mixture Model (GMM) clustering are again analyzed using cluster scatter plots and silhouette scores, similar to the K-means clustering section (see section 3.3). On the Digits dataset, GMM performed the best at 2 clusters; however, the data points were still overlapping and lacking separation. An interesting observation is that even though the Digits dataset contains 10 classes (0-9), the clusters do not visually correspond to 10 clearly separated groups. 


For the Make Blobs dataset, GMM performed the best at 3 clusters, even though you can see 4 clusters clearly visible on the cluster mapping. When the number of clusters is enhanced to 4, GMM separates a clearly defined cluster into 2 groups rather than representing the 4 natural groups. The GMM clustering algorithm still performed better on the Make Blobs dataset in comparison to the Digits dataset.

## 5) References
1.	Akinkugbe, Ayo. “Build a Principal Component Analysis (PCA) Algorithm from Scratch”. Medium. April 2024. https://medium.com/technological-singularity/build-a-principal-component-analysis-pca-algorithm-from-scratch-7515595bf08b

2.	Lakshimi, B Prathiba. “Clustering GMM From Scratch”. Kaggle. 2023. https://www.kaggle.com/code/bprathibalakshmi/clustering-gmm-from-scratch

3.	Luke, Turner. “Create a K-Means Clustering Algorithm from Scratch in Python”. April 2022. https://towardsdatascience.com/create-your-own-k-means-clustering-algorithm-in-python-d7d4c9077670/

4.	Tam, Adrian. “Principal Component Analysis for Visualization”. October 2021. https://machinelearningmastery.com/principal-component-analysis-for-visualization/

5.	Yehoshua, Roi. “Gaussian Mixture Models (GMMs): from Theory to Implementation”. November 2023. https://towardsdatascience.com/gaussian-mixture-models-gmms-from-theory-to-implementation-4406c7fe9847/

6.	“K means Clustering – Introduction”. Geek for Geeks. August 2026. https://www.geeksforgeeks.org/machine-learning/k-means-clustering-introduction/

7.	“Principal Component Analysis (PCA)”. Geek for Geeks. July 2026. https://www.geeksforgeeks.org/data-analysis/principal-component-analysis-pca/

8.	“Proportion of Variance”. Geek for Geeks. July 2025. https://www.geeksforgeeks.org/machine-learning/proportion-of-variance/

9.	“Selecting the number of clusters with silhouette analysis on KMeans clustering”. Scikit learn. https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html







