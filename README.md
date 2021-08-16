  SVM PROJECT

About this Dataset-


Gender Recognition by Voice and Speech Analysis

This database was created to identify a voice as male or female, based upon acoustic properties of the voice and speech. The dataset consists of 3,168 recorded voice samples, collected from male and female speakers. The voice samples are pre-processed by acoustic analysis in R using the seewave and tuneR packages, with an analyzed frequency range of 0hz-280hz (human vocal range).

The Dataset

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
