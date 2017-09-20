# CarND-T3-Prediction-NaiveBayes
Udacity CarND Term3 3. Prediction exercise implementing naive Bayes. 

# Status report:
```
 X_train number of elements 750
 X_train element size 4
 Y_train number of elements 750
 X_test number of elements 250
 X_test element size 4
 Y_test number of elements 250
 **You got 84.80000 correct**
```

# Features extracted:
- Feature 1: even occuring probability:
```
 p_left=0.28533, p_keep=0.42133, p_right=0.29333
```
- Feature 2: offset from road center. Notice keep deviation is really small besides its small offset.
```
 left_d_offset_ave=0.73940, keep_d_offset_ave=0.17360, right_d_offset_ave=0.71776
 left_d_offset_dev=0.30399, keep_d_offset_dev=0.09271, right_d_offset_dev=0.39838
```
- Feature 3: d_dot be negative/left, almost zero/keep, positive/right.
```
 left_d_dot_ave=-0.96709, keep_d_dot_ave=0.00581, right_d_dot_ave=0.95402
 left_d_dot_dev=0.43994, keep_d_dot_dev=0.02827, right_d_dot_dev=0.41841
```

# Theory
- Prediction can be done using Gaussian Naive Bayes method.
* Wikipedia has a very good example of sex classification here:
   https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Gaussian_naive_Bayes
* The theoretcal explanation is also good:
   https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Parameter_estimation_and_event_models

- Use Gaussian distribution to emulate the probability. Need to get u and sigma of each feature.
```
    //      u = average over all same labelled data
    //      sigma^2 = sum((x_i - u)^2)/N
```
