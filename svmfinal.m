% ************************************************************************
%                   SVM-Model-Selection
% ************************************************************************

% This script performs SVM Hyper-Parameter Optimization to search for the 
% best combination of values for hyper-parameters for the given class 
% imbalance problem. It studies SVM with Linear, polynomial and RBF kernels

%% Read the dataset
Daten = readtable('C:\Users\Prerna Prakash Gupta\Desktop\nndata5000.xlsx');
Daten=table2array(Daten)
[m,n] = size(Daten) ;
q=Daten(:,end-1)

%% Partition the data into stratified train and straified test set (class imbalance), with 30% test hold-out dataset
train_test_stratified=cvpartition(q,'HoldOut',0.3);
traindata=Daten(train_test_stratified.training,:);
testdata=Daten(train_test_stratified.test,:);
xtest=testdata(:,1:end-1);
ytest=testdata(:,end);
ytest=vertcat(ytest')

%% Create stratified dataset 5-fold cross-validation from the 70% training data
K=5;
cv_partitions=cvpartition(ytrain,'KFold',K);
validation_performance=[];
ytest=vec2ind(ytest)
xtrain=traindata(:,1:end-1);
ytrain=traindata(:,end);
Scores_EB=[]
[rows_train,cols_train]=size(xtrain);
validation_f_score=[];
prectrain=[]
recalltrain=[]
mcv=[]

%% RBF Kernel: Hyper Parameter Tuning: Grid Search
%G=Gamma value for RBF Kernel
%C=Box constraint for RBF Kernel
G = [0.1, 1, 10, 100];
C = [0.1, 1, 10, 50];

%In order to evaluate the dispersion of Test F1 score, every hyper-parameter combination is evaluated on the 30% Hold-Out test set
tStart = tic;

for ii = 1:length(G)
      for jj = 1: length(C)
      %K-fold Cross validation for different values of C and Gamma          
          for i=1:K
              cv_train_index=cv_partitions.training(i);
              cv_validation_index=cv_partitions.test(i);
              cv_train_x=xtrain(cv_train_index,:);
              cv_train_y=ytrain(cv_train_index,:);
              cv_validation_y=ytrain(cv_validation_index,:);
              cv_validation_x=xtrain(cv_validation_index,:);
              % Model fit on training data and evaluated on validation data for every fold
              net = fitcsvm( cv_train_x,cv_train_y,'KernelFunction', 'rbf', 'BoxConstraint', C(jj), 'KernelScale', G(ii));
              predicted_validation=predict(net, cv_validation_x);
              misclassval=sum(predicted_validation-cv_validation_y)/length(predicted_validation)
              confMat = confusionmat(cv_validation_y, predicted_validation);
              %Confusion Matrix, F1 -score, misscallification erorr;Precision and Recall for every out of sample validation fold
              for j =1:size(confMat,1)
                  recall(j)=confMat(j,j)/sum(confMat(j,:));
              end
              recall(isnan(recall))=[];
              Recall=sum(recall)/size(confMat,1);
              for j =1:size(confMat,1)
                  precision(j)=confMat(j,j)/sum(confMat(:,j));
              end
              precision(isnan(precision))=[];
              Precision=sum(precision)/size(confMat,1)
              prectrain=[prectrain,Precision ]
              recalltrain=[recalltrain,Recall ]
              Fscorevalidation=2*Recall*Precision/(Precision+Recall);            
              validation_f_score=[validation_f_score; Fscorevalidation]; 
              mcv=[mcv,misclassval]
          end
          % Calculating the mean F1 -score, misscallification erorr;Precision and Recall for all out-of sample validation scores to get cross-validation metrics
          validation_f_score=mean(validation_f_score)
          prectrainval=mean(prectrain)
          reaclltrainval=mean(recalltrain)
          MCV= mean(mcv) 
          %Time calculation post K-fold cross validation for the given hyperparameter combination
          tElapsed = toc(tStart);
          
          %Evaluate every-hyper parameter-combination on hold-out test data to study dispersion of test scores
          ytestpredicted=predict(net, xtest);
          MCT= sum(ytestpredicted-ytest)            
          confMat1 = confusionmat(ytest, ytestpredicted);
          for j1 =1:size(confMat1,1)
              recall(j1)=confMat1(j1,j1)/sum(confMat1(j1,:));
          end
          recall(isnan(recall))=[];
          Recall=sum(recall)/size(confMat1,1);

                           
          for j1 =1:size(confMat1,1)
              precision(j1)=confMat1(j1,j1)/sum(confMat1(:,j1));
          end
          precision(isnan(precision))=[];
          Precision=sum(precision)/size(confMat1,1)
          Fscoretest=2*Recall*Precision/(Precision+Recall);
          
          Scores_EB = [Scores_EB; G(ii), C(jj),  MCV, MCT,tElapsed,prectrainval, reaclltrainval,  validation_f_score,Fscoretest, Recall,Precision]
                  
      end
end
%save the grid-seach
writematrix(Scores_EB,'C:\Users\Prerna Prakash Gupta\Desktop\gaussPOLYSVMFINALPPG.csv')

%% Hyper-Parameter Tuning for Polynomial Kernel
%%%G=Polynomial Degree Value
%C=Box constraint for RBF Kernel

G=[2,3,4,5,6,7,8]
C = [0.1, 1, 10, 50];
%In order to evaluate the dispersion of Test F1 score, every hyper-parameter combination is evaluated on the 30% Hold-Out test set
tStart = tic;
for ii = 1:length(G)
      for jj = 1: length(C)
          %K-fold cross validation for different values of C and Polynimal
          %degree
          for i=1:K
              cv_train_index=cv_partitions.training(i);
              cv_validation_index=cv_partitions.test(i);
              cv_train_x=xtrain(cv_train_index,:);
              cv_train_y=ytrain(cv_train_index,:);
              cv_validation_y=ytrain(cv_validation_index,:);
              cv_validation_x=xtrain(cv_validation_index,:);
              net = fitcsvm( cv_train_x,cv_train_y,'KernelFunction', 'polynomial', 'BoxConstraint', C(jj), ...
            'PolynomialOrder', ii);
              % Model fit on training data and evaluated on validation data for every fold
              predicted_validation=predict(net, cv_validation_x);
              
              %Confusion Matrix, F1 -score, misscallification erorr;Precision and Recall for every out of sample validation fold
              misclassval=sum(predicted_validation-cv_validation_y)/length(predicted_validation)
              confMat = confusionmat(cv_validation_y, predicted_validation);
              for j =1:size(confMat,1)
                  recall(j)=confMat(j,j)/sum(confMat(j,:));
              end
              recall(isnan(recall))=[];
              Recall=sum(recall)/size(confMat,1);

                                    % Precision
              for j =1:size(confMat,1)
                  precision(j)=confMat(j,j)/sum(confMat(:,j));
              end
              precision(isnan(precision))=[];
              Precision=sum(precision)/size(confMat,1)
              prectrain=[prectrain,Precision ]
              recalltrain=[recalltrain,Recall ]
              Fscorevalidation=2*Recall*Precision/(Precision+Recall);
                          
              validation_f_score=[validation_f_score; Fscorevalidation]; 
              mcv=[mcv,misclassval]
              tElapsed = toc(tStart);
          end
          % Calculating the mean F1 -score, misscallification erorr;Precision and Recall for all out-of sample validation scores to get cross-validation metrics
          validation_f_score=mean(validation_f_score)
          prectrainval=mean(prectrain)
          reaclltrainval=mean(recalltrain)
          MCV= mean(mcv) 
          %Time calculation post K-fold cross validation for the given hyperparameter combination
          tElapsed = toc(tStart);
          
          
          %Evaluate every-hyper parameter-combination on hold-out test data to study dispersion of test scores
          ytestpredicted=predict(net, xtest);
          MCT= sum(ytestpredicted-ytest)            
          confMat1 = confusionmat(ytest, ytestpredicted);
          for j1 =1:size(confMat1,1)
              recall(j1)=confMat1(j1,j1)/sum(confMat1(j1,:));
          end
          recall(isnan(recall))=[];
          Recall=sum(recall)/size(confMat1,1);

                                    % Precision
          for j1 =1:size(confMat1,1)
              precision(j1)=confMat1(j1,j1)/sum(confMat1(:,j1)); 
          end
          precision(isnan(precision))=[];
          Precision=sum(precision)/size(confMat1,1)
          Fscoretest=2*Recall*Precision/(Precision+Recall);
          Scores_EB = [Scores_EB; G(ii), C(jj),  MCV, MCT,tElapsed,prectrainval, reaclltrainval,  validation_f_score,Fscoretest, Recall,Precision]
                  
      end
end
%save the script of hyper-parameter combinations
writematrix(Scores_EB,'C:\Users\Prerna Prakash Gupta\Desktop\ppg12POLYSVMFINALPPG.csv')
%% Fit Linear Kernel
 
xtest=Daten(:,1:end-1);
ytest=Daten(:,end);
C = [0.25, 0.5, 0.75, 1];
     
for jj = 1: length(C)
    tStart = tic;
    for i=1:K
      cv_train_index=cv_partitions.training(i);
      cv_validation_index=cv_partitions.test(i);
      cv_train_x=xtrain(cv_train_index,:);
      cv_train_y=ytrain(cv_train_index,:);
      cv_validation_y=ytrain(cv_validation_index,:);
      cv_validation_x=xtrain(cv_validation_index,:);
      net = fitcsvm( cv_train_x,cv_train_y,'KernelFunction', 'linear', 'BoxConstraint', C(jj));

       predicted_validation=predict(net, cv_validation_x);
      misclassval=sum(predicted_validation-cv_validation_y)/length(predicted_validation)
      confMat = confusionmat(cv_validation_y, predicted_validation);
      for j =1:size(confMat,1)
          recall(j)=confMat(j,j)/sum(confMat(j,:));
      end
      recall(isnan(recall))=[];
      Recall=sum(recall)/size(confMat,1);

                            % Precision
      for j =1:size(confMat,1)
          precision(j)=confMat(j,j)/sum(confMat(:,j));
      end
      precision(isnan(precision))=[];
      Precision=sum(precision)/size(confMat,1)
      prectrain=[prectrain,Precision ]
      recalltrain=[recalltrain,Recall ]
      Fscorevalidation=2*Recall*Precision/(Precision+Recall);

      validation_f_score=[validation_f_score; Fscorevalidation]; 
      mcv=[mcv,misclassval]
      tElapsed = toc(tStart);
  end

  validation_f_score=mean(validation_f_score)
  prectrainval=mean(prectrain)
  reaclltrainval=mean(recalltrain)
  MCV= mean(mcv) 




  ytestpredicted=predict(net, xtest);
  MCT= sum(ytestpredicted-ytest)            
  confMat1 = confusionmat(ytest, ytestpredicted);
  for j1 =1:size(confMat1,1)
      recall(j1)=confMat1(j1,j1)/sum(confMat1(j1,:));
  end
  recall(isnan(recall))=[];
  Recall=sum(recall)/size(confMat1,1);

                            % Precision
  for j1 =1:size(confMat1,1)
      precision(j1)=confMat1(j1,j1)/sum(confMat1(:,j1)); 
  end
  precision(isnan(precision))=[];
  Precision=sum(precision)/size(confMat1,1)
  Fscoretest=2*Recall*Precision/(Precision+Recall);
  Scores_EB = [Scores_EB;  C(jj),  MCV, MCT,tElapsed,prectrainval, reaclltrainval,  validation_f_score,Fscoretest, Recall,Precision]


end
writematrix(Scores_EB,'C:\Users\Prerna Prakash Gupta\Desktop\12POLYSVMFINALPPG.csv')