# UnFold_ML_project
######## CREDITS
# ML for science Project for UNFoLD Lab
# Birch,H.; Von Sury,B.; Vuillecard, P.;
# 2019-10 ~ 2019-12

######## DISCLAIMER
# This code is provided as-is, the autors are not responsible for any (miss-)use, etc.

###### SOURCES
https://scikit-learn.org/stable/install.html
https://www.tensorflow.org/install/pip
https://www.pyimagesearch.com/2016/11/14/installing-keras-with-tensorflow-backend/

######################################READ-ME##############################################

###### SET UP
Before running our code please install Scikit learn and Keras using the following instructions:
It is recommended to create a virtual environment when doing a project but we won't go into details here 
as it is not relevant but more information can be found here:

https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/
# installing Scikit:


In your command prompt execute the following code:

pip install -U scikit-learn

To check if it has been correctly installed execute :

python -m pip show scikit-learn # to see which version and where scikit-learn is installed
python -m pip freeze # to see all packages installed in the active virtualenv
python -c "import sklearn; sklearn.show_versions()"

# installing Keras and Tensorflow:

Warning: They are different version of Tensorflow depending on whether you have a GPU or not!

In your command prompt execute the following code:

pip install --upgrade tensorflow

To check if it has been correctly installed execute :

python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

In your command prompt execute the following code:

pip install keras

#### NOTEBOOKS AND CODE 

All the results that we have in the paper can be find in the notebooks for reproducibility.
All the code is pre run to show all the plots and graphes. 
There is also file .py that contain all the function that we have use in all the notebooks.

#NOTEBOOKS:
-Kinematic_prediction : contain all result of the part 4 of the paper. 
-NN_CL : contain all the neural networks that we have tryed to predict the lift coefficient 
-NN_CP : contain all the neural network that we have tryed to predict the power coefficient 
-NN_mean_CL : contain all the neural network that we have tryed to predict the average lift coefficient 
-NN_mean_CP : contain all the neural network that we have tryed to predict the average power coefficient 
-Regression_average_CP_CL : contain all the linear regression that we have tryed to predict the average power coefficient and average lift coefficient
-Regression_CP_CL : contain all the linear regression that we have tryed to predict the power coefficient and lift coefficient

#CODE:
-helpers.py
-kinematic.py
-neural_networks.py
-regression
-NN_mean

#data
-ml_data : contain the data of kinematic , lift coefficient, power coefficient
-CP_pred : contain the prediction of power coefficient from kinematic of the pareto front
-CL_pred : contain the prediction of power coefficient from kinematic of the pareto front
