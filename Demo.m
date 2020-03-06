clear all
clc
close all
%% Import Training Data
load ExampleTrainingData
Label_Training=data(:,1);
Data_Training=data(:,2:1:6);

%% Import Testing Data
load ExampleTestingData
Label_Testing=data(:,1);
Data_Testing=data(:,2:1:6);

%% Use SASYNO to Create Synthetic Data and Address the Problem of Class Imbalance
[Augmented_Data_Training,Augmented_Label_Training]=SASYNO(Data_Training,Label_Training);

%% Use the Augmented Training Data for Decison Tree Classifier Training
mdl = fitctree(Augmented_Data_Training,Augmented_Label_Training);

%% Predict the Class Labels of Unlabelled Testing Data
Label_Prediction = predict(mdl,Data_Testing);

%% Calculate the Confusion Matrix
ConfusionMatrix=confusionmat(Label_Testing,Label_Prediction)