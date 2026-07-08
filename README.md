# Offline Signature Verification (Digital Image Processing)

## Project Description
An offline signature verification system based on Digital Image Processing (DIP) using Python. This algorithm exploration project was developed entirely using a conventional approach (static mathematical computation and heuristics) without the use of Machine Learning. 

The main focus of this development is the implementation of data structures and the understanding of advanced programming logic for image matrix processing, spatial feature extraction, and vector similarity metric measurement.

## Dataset
Testing and performance evaluation of this system utilize the **GPDS Signature Dataset**, which is a common reference standard in signature verification research (containing genuine signature samples and forgery scenarios). This dataset can be accessed publicly via the following link:
* [Kaggle: GPDS-1150 Dataset](https://www.kaggle.com/datasets/adeelajmal/gpds-1150)

## Processing Pipeline
1. **Pre-processing:** *Separable Gaussian Blur* (noise reduction), *Manual Adaptive Thresholding* (local average-based), and *Custom Kernel Sharpening* (line enhancement).
2. **Feature Extraction:** *Auto-Crop Bounding Box*, *Square Padding* (distortion avoidance), and pixel density ratio extraction based on a 15x15 *Grid*.
3. **Verification:** Feature vector similarity measurement using *Cosine Similarity* with 1-to-1 inference against a *Static Threshold*.

## Performance Evaluation and Trade-Off Analysis
At the optimal equilibrium configuration (15x15 Grid with `SIMILARITY_THRESHOLD = 72.0`), the system achieved a peak accuracy of **78.06%**. 

The absence of dynamic weighting from Machine Learning in this system actually opens up room to dissect the fundamental limitations of static inference algorithms. This is clearly visible through the trade-off phenomenon between the False Reject Rate (FRR) and False Accept Rate (FAR) from a total of 3300 test samples:

* **FRR (False Reject Rate): 38.83%** 
  The system incorrectly rejected 466 out of 1200 genuine signature files.
* **FAR (False Accept Rate): 12.29%** 
  The system incorrectly accepted 258 out of 2100 forged signature files.

### Security Priority: Why is FRR Higher than FAR?
The metric characteristics above indicate that this system is quite "paranoid". This static inference design consciously chooses to sacrifice FRR to suppress the FAR figure as low as possible. 

In real-world security system architectures or document authentication, **the risk of FAR is far more fatal and detrimental compared to FRR**. Approving a forged signature (FAR) means providing a legitimate security loophole to unauthorized parties, potentially leading to material loss or data breaches. Conversely, rejecting a genuine signature (FRR) only causes minor operational inconvenience, where the legitimate user is simply asked to re-sign with a more precise stroke. Therefore, the system is maintained at the 72.0 threshold as a solid defense against forgery attacks.

### Conclusion
The 78.06% performance metric is not merely a number of system limitations, but an analytical manifestation of the structural compromise of static algorithms. The system's ability to detect anomalies (FAR 12.29%) proves that conventional feature extraction logic still possesses structural reliability. This trade-off analysis affirms that good software architecture design demands a comprehensive understanding of risk profiles, while also serving as a rational foundation for the urgency of transitioning towards deep learning architectures for future feature adaptability.