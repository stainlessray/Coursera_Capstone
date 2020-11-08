## About the Methods
<p>
To begin we looked at the overall distribution of accidents throughout the state. The most populous of Delaware's three counties is Newcastle. It is also a major transportation hub for interstate travel because it hosts a very important section of the I-95 corrider stretching almost the entire length of the East Coast.<br>

 The distribution of severity grades give way to some new inferences about "Conditions" as a predictor. We found that the overall dimension of the dataset is too great. So when all of those data are encoded and fit for modeling, the propensity for overfit jumps out at you. Combining a single "Specific" or non-general statistic was effective at taming the model, but nowhere near accurate enough to be useful in prediction. So we looked at it from another angle to determine if there was any corroboration for the initial take that - This data is too detailed!<br>
 
 We tested with a broad base of prediction algorithms. Some of them were: Support Vector Machine (SVM), Decision Tree, Logistic Regression, and finally K Nearest Neighbor classification (KNN). We adopted KNN in the end because it was much more reliable in reproduction, and it provided the best tolerance for a single dominant feature in the Severity grade distribution. The second tier severity quantifies accidents that are individually, not very impacftful on road conditions. However, the tier two accident represents well over 70% of the datapoints in the sample. Why is that a problem? Looking at the predictions, several of the models took the easy way to "high accuracy". They just predicted tier two exclusively. This lead to the appearance of extremely high accuracy and extremely low logloss. It wasn't predicting anything. It decided not to play the game! <br>
 
 
</p>
