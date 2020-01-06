# SMS Spam Filter Training Command Line App

Training and testing a model on a dataset of sms messages.
<p>
  <b>Default Arguments:</b>
<br>
Random Forest Classification (150 estimators, no max-depth)
<br>
TfIdf Vectorizer
<br>
Stemming Pre-Processing
<br>
SMSSpamCollection.txt dataset
</p>
<p>
  <b>Optional arguments include:</b>
<br>
-r -Random Forest Classification
<br>
-g -Gradient Boosting Classification
<br>
-n -Number of Estimators (Classification Parameter)
<br>
-m -Max Depth (Classification Parameter)
<br>
-t -TfIdf Vectorization
<br>
-c -Count Vectorization (1 gram)
<br>
-s -Stemming SMS Content When Pre-Processing
<br>
-l -Lemmitizing SMS Content When Pre-Processing
<br>
-d -Dataset Directory
</p>
