# SVM-for-a-classification-problem-
MATLAB Script:
• Load "SVMFinal.mat"
• Read csv file: NN5000.csv containing dataset
• Partition the data into stratified train and stratified test set (class imbalance), with 30% test hold-out dataset.
• On the 70% training data, perform stratified k-Fold cross validation for model selection.
• ****%% Hyper-Parameter Tuning for RBF Kernel************
• G=Gamma Value
• C=Box constraint for RBF Kernel
• start measuring computation time
• K-fold Cross validation for different values of C and Gamma
• Model fit on training data and evaluated on validation data for every fold
• Confusion Matrix, F1 -score, misclassification erorr; Precision and Recall for every out of sample validation fold
• Calculating the mean F1 -score, misclassification erorr;Precision and Recall for all out-of sample validation scores to
get cross-validation metrics
• Time calculation post K-fold cross validation for the given hyperparameter combination
• Evaluate every-hyper parameter-combination on hold-out test data to study dispersion of test scores\
• Save the grid-search in a csv file
• **********%%Hyper-Parameter Tuning for Polynomial Kernel%%***********
• G=Polynomial Degree Value
• C=Box constraint for RBF Kernel
• start measuring computation time
• K-fold Cross validation for different values of C and Gamma
• Model fit on training data and evaluated on validation data for every fold
• Confusion Matrix, F1 -score, misclassification erorr; Precision and Recall for every out of sample validation fold
• Calculating the mean F1 -score, misclassification on erorr; Precision and Recall for all out-of sample validation scores
to get cross-validation metrics
• Time calculation post K-fold cross validation for the given hyperparameter combination
• Evaluate every-hyper parameter-combination on hold-out test data to study dispersion of test scores
• Save the grid-search in a csv file
