__SVM PROJECT__

Support Vector Machines Classifier Tutorial with Python

Support Vector Machines (SVMs in short) are supervised machine learning algorithms that are used for classification and regression purposes. In this kernel, I build a Support Vector Machines classifier to classify a Pulsar star. I have used the Predicting a Pulsar Star dataset for this project.


__Table of Contents__

___1.Introduction to Support Vector Machines___

Support Vector Machines (SVMs in short) are machine learning algorithms that are used for classification and regression purposes. SVMs are one of the powerful machine learning algorithms for classification, regression purposes. An SVM classifier builds a model that assigns new data points to one of the given categories. Thus, it can be viewed as a non-probabilistic binary linear classifier.

The original SVM algorithm was developed by Vladimir N Vapnik and Alexey Ya. Chervonenkis in 1963. At that time, the algorithm was in early stages. The only possibility is to draw hyperplanes for linear classifier. In 1992, Bernhard E. Boser, Isabelle M Guyon and Vladimir N Vapnik suggested a way to create non-linear classifiers by applying the kernel trick to maximum-margin hyperplanes. The current standard was proposed by Corinna Cortes and Vapnik in 1993 and published in 1995.

SVMs can be used for linear classification purposes. In addition to performing linear classification, SVMs can efficiently perform a non-linear classification using the kernel trick. It enable us to implicitly map the inputs into high dimensional feature spaces.

___2.Support Vector Machines intuition___

__Hyperplane__

A hyperplane is a decision boundary which separates between given set of data points having different class labels. The SVM classifier separates data points using a hyperplane with the maximum amount of margin. This hyperplane is known as the maximum margin hyperplane.

__Support Vectors__

Support vectors are the sample data points, which are closest to the hyperplane.
It helps to determine maximum distance space.

__Margin__

A margin is a separation gap between the two lines on the closest data points. It is calculated as the perpendicular distance from the line to support vectors or closest data points. In SVMs, we try to maximize this separation gap so that we get maximum margin.

The following diagram illustrates these concepts visually.


__Margin in SVM__

![image](https://user-images.githubusercontent.com/89013703/129720649-930721f9-03fa-459c-aca8-2802228a8be4.png)

SVM Under the hood

In SVMs, our main objective is to select a hyperplane with the maximum possible margin between support vectors in the given dataset. SVM searches for the maximum margin hyperplane in the following 2 step process –

Generate hyperplanes which segregates the classes in the best possible way. There are many hyperplanes that might classify the data. We should look for the best hyperplane that represents the largest separation, or margin, between the two classes.

So, we choose the hyperplane so that distance from it to the support vectors on each side is maximized. If such a hyperplane exists, it is known as the maximum margin hyperplane and the linear classifier it defines is known as a maximum margin classifier.

The following diagram illustrates the concept of maximum margin and maximum margin hyperplane in a clear manner.

__Maximum margin hyperplane__

![image](https://user-images.githubusercontent.com/89013703/129721106-744d82e3-5cdb-4bac-a5c2-52427776b08e.png)

___3.Kernel trick___

In practice, 
SVM algorithm is implemented using a kernel. It uses a technique called the kernel trick. In simple words, a kernel is just a function that maps the data to a higher dimension where data is separable. A kernel transforms a low-dimensional input data space into a higher dimensional space. So, it converts non-linear separable problems to linear separable problems by adding more dimensions to it. Thus, the kernel trick helps us to build a more accurate classifier. Hence, it is useful in non-linear separation problems.

__Problem with dispersed datasets__

Sometimes, the sample data points are so dispersed that it is not possible to separate them using a linear hyperplane. In such a situation, SVMs uses a kernel trick to transform the input space to a higher dimensional space as shown in the diagram below. It uses a mapping function to transform the 2-D input space into the 3-D input space. Now, we can easily segregate the data points using linear separation.

___Kernel trick - transformation of input space to higher dimensional space___

![image](https://user-images.githubusercontent.com/89013703/129721480-7d35f25a-6bcf-40ca-9649-bf516a13dd96.png)

Kernel function

![image](https://user-images.githubusercontent.com/89013703/129722009-ee1d6fb2-adfc-4fbb-b83f-50e214f15dd4.png)

In the context of SVMs, there are 4 popular kernels – Linear kernel,Polynomial kernel,Radial Basis Function (RBF) kernel (also called Gaussian kernel) and Sigmoid kernel. 
These are described below -

__Linear kernel__

Linear kernel is used when the data is linearly separable. It means that data can be separated using a single line. It is one of the most common kernels to be used. It is mostly used when there are large number of features in a dataset. Linear kernel is often used for text classification purposes.

Training with a linear kernel is usually faster, because we only need to optimize the C regularization parameter. When training with other kernels, we also need to optimize the γ parameter. So, performing a grid search will usually take more time.

Linear kernel can be visualized with the following figure.

![image](https://user-images.githubusercontent.com/89013703/129722853-ab6c3add-045f-4feb-a086-b97e1a1afdee.png)

__Polynomial Kernel__

Polynomial kernel represents the similarity of vectors (training samples) in a feature space over polynomials of the original variables. The polynomial kernel looks not only at the given features of input samples to determine their similarity, but also combinations of the input samples.

Polynomial kernel is very popular in Natural Language Processing. The most common degree is d = 2 (quadratic), since larger degrees tend to overfit on NLP problems. 
It can be visualized with the following diagram.

![image](https://user-images.githubusercontent.com/89013703/129723398-2126e32b-1602-4b76-a4d8-fa1b2531c4e1.png)

__Radial Basis Function Kernel__
Radial basis function kernel is a general purpose kernel. It is used when we have no prior knowledge about the data.

SVM Classification with rbf kernel

![image](https://user-images.githubusercontent.com/89013703/129723672-bd8023b6-fb9b-4a22-b589-01fa9e709347.png)

__Sigmoid kernel__
Sigmoid kernel has its origin in neural networks. We can use it as the proxy for neural networks.

Sigmoid kernel      ![image](https://user-images.githubusercontent.com/89013703/129723825-d1e65e7e-072b-4850-b677-866dfd24d5fe.png)


___4.SVM Scikit-Learn libraries___

Scikit-Learn provides useful libraries to implement Support Vector Machine algorithm on a dataset. There are many libraries that can help us to implement SVM smoothly. We just need to call the library with parameters that suit to our needs. In this project, I am dealing with a classification task. So, I will mention the Scikit-Learn libraries for SVM classification purposes.

First, there is a LinearSVC() classifier. As the name suggests, this classifier uses only linear kernel. In LinearSVC() classifier, we don’t pass the value of kernel since it is used only for linear classification purposes.

Scikit-Learn provides two other classifiers - SVC() and NuSVC() which are used for classification purposes. These classifiers are mostly similar with some difference in parameters. NuSVC() is similar to SVC() but uses a parameter to control the number of support vectors. We pass the values of kernel, gamma and C along with other parameters. By default kernel parameter uses rbf as its value but we can pass values like poly, linear, sigmoid or callable function.

___5.Dataset description___

__About this Dataset__

I am using Gender Recognition by Voice and Speech Analysis for this Project

This database was created to identify a voice as male or female, based upon acoustic properties of the voice and speech. The dataset consists of 3,168 recorded voice samples, collected from male and female speakers. The voice samples are pre-processed by acoustic analysis in R using the seewave and tuneR packages, with an analyzed frequency range of 0hz-280hz (human vocal range).

__Attribute Information__

The following acoustic properties of each voice are measured and included within the CSV:


1)meanfreq: mean frequency (in kHz)

2)sd: standard deviation of frequency

3)median: median frequency (in kHz)

4)Q25: first quantile (in kHz)

5)Q75: third quantile (in kHz)

6)IQR: interquantile range (in kHz)

7)skew: skewness (see note in specprop description)

8)kurt: kurtosis (see note in specprop description)

9)sp.ent: spectral entropy

10)sfm: spectral flatness

11)mode: mode frequency

12)centroid: frequency centroid (see specprop)

13)meanfun: average of fundamental frequency measured across acoustic signal

14)minfun: minimum fundamental frequency measured across acoustic signal

15)maxfun: maximum fundamental frequency measured across acoustic signal

16)meandom: average of dominant frequency measured across acoustic signal

17)mindom: minimum of dominant frequency measured across acoustic signal

18)maxdom: maximum of dominant frequency measured across acoustic signal

19)dfrange: range of dominant frequency measured across acoustic signal

20)modindx: modulation index. Calculated as the accumulated absolute difference between adjacent measurements of fundamental frequencies divided by the frequency range

21)label: male or female


___6.Import libraries___

___7.Import dataset___

___8.Exploratory data analysis___

___9.Declare feature vector and target variable___

___10.Feature Scaling___

___11.Split data into separate training and test set___

___12.Run SVM with default hyperparameters___

___13.Check for overfitting and underfitting___

___14.ROC-AUC Curve___

___15.Calculate Cross-Validation score___

___16.KFOld CV with Shuffle___

___17.Hyperparameter optimization using GridSearch CV

___18.Results and conclusion___

There are outliers in our dataset. So, as I increase the value of C to limit fewer outliers, the accuracy increased. This is true with different kinds of kernels.

We get maximum accuracy with rbf and linear kernel with C=100.0 and the accuracy is 0.9832. So, we can conclude that our model is doing a very good job in terms of predicting the class labels. But, this is not true. Here, we have an imbalanced dataset. Accuracy is an inadequate measure for quantifying predictive performance in the imbalanced dataset problem. So, we must explore confusion matrix that provide better guidance in selecting models.

ROC AUC of our model is very close to 1. So, we can conclude that our classifier does a good job in classifying the pulsar star.

I obtain higher average stratified k-fold cross-validation score of 0.9789 with linear kernel but the model accuracy is 0.9832. So, stratified cross-validation technique does not help to improve the model performance.

Our original model test accuracy is 0.9832 while GridSearch CV score on test-set is 0.9835. So, GridSearch CV helps to identify the parameters that will improve the performance for this particular model.



___19.References___

So, now we will come to the end of this Project.
Thank You.
