__SVM PROJECT__

Support Vector Machines Classifier Tutorial with Python

Support Vector Machines (SVMs in short) are supervised machine learning algorithms that are used for classification and regression purposes. In this kernel, I build a Support Vector Machines classifier to classify a Pulsar star. I have used the Predicting a Pulsar Star dataset for this project.


__Table of Contents__

___1.Introduction to Support Vector Machines___

Support Vector Machines (SVMs in short) are machine learning algorithms that are used for classification and regression purposes. SVMs are one of the powerful machine learning algorithms for classification, regression purposes. An SVM classifier builds a model that assigns new data points to one of the given categories. Thus, it can be viewed as a non-probabilistic binary linear classifier.

The original SVM algorithm was developed by Vladimir N Vapnik and Alexey Ya. Chervonenkis in 1963. At that time, the algorithm was in early stages. The only possibility is to draw hyperplanes for linear classifier. In 1992, Bernhard E. Boser, Isabelle M Guyon and Vladimir N Vapnik suggested a way to create non-linear classifiers by applying the kernel trick to maximum-margin hyperplanes. The current standard was proposed by Corinna Cortes and Vapnik in 1993 and published in 1995.

SVMs can be used for linear classification purposes. In addition to performing linear classification, SVMs can efficiently perform a non-linear classification using the kernel trick. 

___2.Support Vector Machines intuition___

__Hyperplane__

A hyperplane is a decision boundary which separates between given set of data points having different class labels. The SVM classifier separates data points using a hyperplane with the maximum amount of margin. This hyperplane is known as the maximum margin hyperplane.

__Support Vectors__

Support vectors are the sample data points, which are closest to the hyperplane.
It helps to determine maximum distance space.

__Margin__

A margin is a separation gap between the two lines on the closest data points. It is calculated as the perpendicular distance from the line to support vectors or closest data points. In SVMs, we try to maximize this separation gap so that we get maximum margin.

The following diagram illustrates these concepts visually.


Margin in SVM

![image](https://user-images.githubusercontent.com/89013703/129720649-930721f9-03fa-459c-aca8-2802228a8be4.png)





___3.Kernel trick___

___4.SVM Scikit-Learn libraries___

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

___19.References___

