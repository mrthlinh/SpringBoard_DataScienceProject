## Table of contents


## Part 1: Inferential Statistics

1. What descriptive measure in statistics do you know?

    __Answer:__

  - Mean,median: an average or most commonly indicated response.
  - Standard deviation, variance: How "spread out" the  data are, which one is too large or too small
  - Percentile, Quartile:

    Percentile: You are (1.85m) the fourth tallest person in a group of 20, it mean 80% people are shorter than you, you are at 80th percentile or 1.85m is the 80th percentile.

    Quartile: split data into 4 group, Q1 (lower quartile - 25th percentile), Q2 (middle quartile or median - 50th percentile), Q3 (upper quartile - 75th percentile)

2. What’s the difference between the mean, the median and the mode?

    __Answer:__

|   Mean   |      Median     |      Mode     |
|:--------:|:---------------:|:-------------:|
|Mean is the average number (sum all sample / number of samples)|Median is the middle number, number of position (N/2)th |Mode is the most frequent number in sample|

3. What is the difference between Type I and Type II error?

    __Answer:__

    We usually refer this error in statistical hypothesis testing.

|    Type 1 error    |   Type 2 error     |
|:------------------:|:------------------:|
|False Positive (You say 1 but it is 0 - You reject a true null hypothesis)|False Negative (You say 0 but it is 1 - you accept a false null hypothesis)|

4. What is a p-value, and what is it used for?

    __Answer:__
     (draft) small p-value means we can reject the null hypothesis and accept the alternative hypothesis

5. What are the null and alternative hypothesis in a statistical test?
    __Answer:__

    Null and alternative hypothesis are two mutually exclusive statement. A hypothesis test uses sample data to determine whether to reject the null hypothesis.

  - Null hypothesis: A statement that a population parameter (such as mean, std, proportion) is equal to a hypothesized value.
  - Alternative hypothesis: A statement that a population parameter (such as mean, std, proportion) is less / greater/ different than a hypothesized value. The alternative hypothesis is what we might believe to be true or hope to prove true.


6. What is a t-test? Do you know what is its relationship to the z-test?

    __Answer:__
    t-test and z-test both base on an assumption that the null distribution is Normal Distribution
    |||
    |:---:|:---:|
    |Z-test|Z-test is used when the null distribution of test statistic is Standard Normal -> std = 1|
    |T-test|T-test is used when we don't know std of of the distribution|


7. What is the power of a statistical test?

    __Answer:__

    Probability of rejecting a false null hypothesis is __power of the test__ or Probability to avoid False Negative Error (Type II)

8. What is the standard deviation?

    __Answer__ Standard deviation tells you the variance of the data and which data point can be considered super large or super small (or a.k.a outliers)

9. What is a confidence interval, and what is it used for?

    __Answer__ Confidence interval is an interval that contains estimated parameter with a certain probability. For example 90% Confidence Interval means there is 90% chance that the interval will cover the true parameter.


10. What is bootstrapping/resampling used for?

    __Answer:__ Boostrap is used to estimate parameter by Monte Carlo simulations when it is too difficult to do it analytically. Boostrap bascially resample (choose a subset randomly) original data and estimate the parameter on those resample data.

11. Do you know the difference between frequentist and Bayesian statistics?

    __Answer:__


12. What are outliers? How can you check for them?

    __Answer:__

13. What is a correlation coefficient? What range of values can it take?

    __Answer:__

14. What is the Central Limit Theorem?

    __Answer:__

15. What is the normal distribution? How can you test if data is normally distributed?

    __Answer:__

16. Can you explain the major types of plots - histogram, bar chart, box plot, and scatter plot?

    __Answer:__

    - histogram: present distribution of the data, simply tell you how data are distributed in different ranges.
    - bar chart: represent value in a rectangle width x height, which height is corresponding to the value.
    - Box plot: represent data in a box with two tails Box plot tells you mean value, outliers, Q1, Q2, Q3.
    - scatter plot: show how data points locate in a 2D dimension.

17. What is a correlation matrix?

    __Answer:__ correlation matrix is a matrix of pair-wise correlation between features / predictors

## Part 2: Machine Learning

- Fundamentals
    - Can you make the distinction between an algorithm and a model?

      __Answer:__ Model is a function representing a data set, algorithms are a way to obtain that function

    - What’s the difference between supervised and unsupervised learning? Labeled vs. unlabeled data

      __Answer:__ supervised learning is based on labeled data while in unsupervised learning, data are unlabeled.

    - What’s the difference between a regression and a classification problem? How about clustering?

      __Answer:__ Regression is a supervised problem where the label is a continuous value like stock price, house price. classification is also a supervised problem but the label is a fixed length set like A,B,C,D. Clustering is a unsupervised problem where we want to find some groups of cluster of data.

    - Why do we use a train/test split?

      __Answer:__ Avoid overfitting and better generalization

    - What is cross-validation used for? What types of cross-validation do you know?

      __Answer:__ If we just fit the data on the training set, we would never use test set at training. It could be a waste of data, consider that data are very valuable. Cross-validation solves this problem by using all data but still keep generalization.

      Type of cross-validation:
      - leave-one-out CV
      - k-fold CV


    - What is generalization error?

      __Answer:__ The error on test set

    - What is the bias-variance trade off?

      __Answer:__ In statistics and machine learning, the bias–variance tradeoff is the property of a set of predictive models whereby models with a lower bias in parameter estimation have a higher variance of the parameter estimates across samples, and vice versa.

    - What is the difference between overfit and underfit?

      __Answer:__

      Overfit is when we use too complex function to capture a simple relation. We could see very high accuracy in training set but very low accuracy in test set.

      Underfit is when we use too simple function to capture a very complex relation. We sould see a low accuracy in both training set and test set.

    - Video: What accuracy metrics do you know, both for classification and for regression? When would you use one metric vs the other?

      __Answer:__


    - What is the curse of dimensionality?

      __Answer:__

    - Why do you need to set the random seed prior to running certain ML algorithms?

      __Answer:__


- Regression
    - Can you explain the difference between Linear and Logistic Regression?

    __Answer__ Linear Regression is for regression problem while logistic regression deals with classification problem.

    - How are the coefficients in a Linear Regression interpreted?

    __Answer__ Coefficient in linear regression can be interpreted as growth rate. For example, in house price
     prediction, coefficient of feature "Area" is 50, it means if "Area" of the house increase by 1, the house price increases by 50.

    - How is the intercept in a Linear Regression interpreted?

    __Answer__ The base value of response (Y) when predictor (X) = 0

    - Can the coefficients in a Logistic Regression be directly interpreted?

    __Answer__ No

    - What is the Adjusted R-Squared? What range of values can it take?

    __Answer__ Adjusted R-Squared is R-squared with penalty in number of features (add more useless features would increase R-squared but decrease Adjusted R-squared), it ranges from 0 to 1

    - Why is the Adjusted R-Squared a better measure than the regular R-Squared?

    __Answer__ Add more useless features would increase R-squared but decrease Adjusted R-squared.

    - How does Logistic Regression work “under the hood”? Can you explain Gradient Descent?

    __Answer__


- SVM
    - Can you explain how a Support Vector Machine works?

    __Answer__ SVM is a soft margin classifier. It means SVM will find a separate line that maximize margin between two side. SVM allows some violations, determine by parameter C, larger C -> more violations and vice versa.

    - What is the kernel trick?
    Adopted https://towardsdatascience.com/understanding-the-kernel-trick-e0bc6112ef78

    __Answer__ In order to maximize the margin, we don't need exact data points X, we only neet its inner product X_transpose * X. So if we transform X into higher dimension (in many cases it can help us classify better), we only need compute its inner products only without actually visiting it. Coool ha?


    - Where does the “support vector” term come in the SVM name?

    - What kind of kernels exist for SVMs?

    __Answer__  linear, polynomial, radical / Gaussian

- Trees
    - How do Decision Trees work?

    - What criteria does a tree-based algorithm use to decide on a split?

    __Answer__

    - How does the Random Forest algorithm work? What are the sources of randomness?


    - How is feature importance calculated by the Random Forest?

    __Answer__

- Other Supervised Learning
    - What is the difference between Bagging and Boosting?

    __Answer__

    - Bagging:
    - Boosting:

    - How do Gradient Boosted Machines work?

    __Answer__

    - What is Regularization, and what types do you know? Avoid overfitting, L1 and L2 regularization. L1 used as dim reduction, L2 better for overall generalization, bonus points for ElasticNet

    __Answer__

    - Can a Random Forest and a GBM be easily parallelized? Why/why not?

    __Answer__

- Unsupervised Learning
    - How does PCA work? What are the uses cases for it?
    - How can you determine the optimal number of principal components?
    - How does the K-Means algorithm work? What are its limitations?
    - What other clustering algorithms do you know?
    - How can you assess the quality of clustering?
    - Can you explain in detail any other clustering algorithms besides K-Means?

__Other__

1. Explain what regularization is and why it is useful.
2. Which data scientists do you admire most? which startups?
3. How would you validate a model you created to generate a predictive model of a quantitative outcome variable using multiple regression.
4. Explain what precision and recall are. How do they relate to the ROC curve?
5. How can you prove that one improvement you've brought to an algorithm is really an improvement over not doing anything?
6. What is root cause analysis?
7. Are you familiar with pricing optimization, price elasticity, inventory management, competitive intelligence? Give examples.
8. What is statistical power?
9. Explain what resampling methods are and why they are useful. Also explain their limitations.
10. Is it better to have too many false positives, or too many false negatives? Explain.
11. What is selection bias, why is it important and how can you avoid it?
12. Give an example of how you would use experimental design to answer a question about user behavior.
13. What is the difference between "long" and "wide" format data?
14. What method do you use to determine whether the statistics published in an article (e.g. newspaper) are either wrong or presented to support the author's point of view, rather than correct, comprehensive factual information on a specific subject?
15. Explain Edward Tufte's concept of "chart junk."
16. How would you screen for outliers and what should you do if you find one?
17. How would you use either the extreme value theory, Monte Carlo simulations or mathematical statistics (or anything else) to correctly estimate the chance of a very rare event?
18. What is a recommendation engine? How does it work?
19. Explain what a false positive and a false negative are. Why is it important to differentiate these from each other?
20. Which tools do you use for visualization? What do you think of Tableau? R? SAS? (for graphs). How to efficiently represent 5 dimension in a chart (or in a video)?
https://www.edureka.co/blog/interview-questions/data-science-interview-questions/
