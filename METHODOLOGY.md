## About the Methods
<p>
To begin we looked at the overall distribution of accidents throughout the state. The most populous of Delaware's three counties is Newcastle. It is also a major transportation hub for interstate travel because it hosts a very important section of the I-95 corrider stretching almost the entire length of the East Coast. Understanding the chaos
inherent to volumous travel scenarios is challenging to say the least. And our dataset, while selected for it's comprehensiveness, has too much dimension taken whole cloth, to be of any benefit. <br>

So when all of those data are encoded and fit for modeling, the propensity for overfit jumps out at you. Combining a single "General" or non-specific datapoint (eg "Zipcode"), was effective at taming the model, but no where near accurate enough to be useful in prediction. The distribution of severity grades also give way to some new inferences about "Conditions" as a predictor. So we bagan our machine learning exploration with the goal to find out what we could do about the initial take that - This data is too detailed!<br>
 
We tested with a broad base of prediction algorithms. Some of them were: Support Vector Machine (SVM), Decision Tree, Logistic Regression, and finally K Nearest Neighbor classification (KNN). We adopted KNN in the end because it was much more reliable in reproduction, and it provided the best tolerance for a single dominant feature in the Severity grade distribution. The second tier severity quantifies accidents that are individually, not very impacftful on road conditions. However, the tier two accident represents over two thirds of the datapoints in the sample. Why is that a problem? Looking at the predictions, several of the models took the easy way to "high accuracy". They just predicted tier two exclusively. This lead to the appearance of extremely high accuracy, but exceedingly high logloss. It wasn't predicting anything. It decided not to play the game!! <br>
 
 
</p>
