The aim of the project is to develop a technical solution by designing a predictive model to help a financial institution to make better decision in accepting or rejecting the loan applications. The model predicts if the applicant will pay off the loan successfully or not. The unique aspects of this project that makes it challenging to do predictive analysis are: (a) the relatively small number of data points and potential predictive features (500 data points, 7 features respectively), and (b) the lack of typical features like credit history, debt to income ratio, properties owned, which are considered as good predictors in these types of prediction problems. The main pipeline implemented in python and included these steps: 

- Data preparation: including amputation of missing values and outliers (identified by IQR ranges), encoding class labels and categorical features, categorization of date data, and finally data standardization (scaling) as a necessary step for algorithms based on some measures of distance like regression.
- Exploratory data analysis and feature selection: looking at grouped distribution of different features across different  levels of target variable (paid-off versus default). Investigation of cross-correlations within features and between features and the target variable. 
- Implementation, validation and training of different binary classifiers: different predictor models (SVM, KNN, logistic regression, random forest, and deep learning) were implemented in this step. I selected proper accuracy measures (F1, Matthew’s correlation coefficient, and area under ROC), used k-fold cross-validation (on training dataset) and tuned the hyper-parameter for each model accordingly to reach the best accuracy while considering the domain knowledge. More specifically minimizing the number of false negatives (to avoid accepting applications from clients who will not pay back their loan) was more important than minimizing the number of false positives (rejecting some good clients). 
- Selecting the best predictor based on their performance on test data. I produced a final visualization graph representing the amount of money the bank will gain or lose if using different models. I also included four baseline cases to make the comparisons easy to understand for stakeholders who are not necessarily interested in details of machine learning algorithms. The baselines included accepting all loan applications, rejecting all applications, using a 50% random guess to accept or reject, and using an ideal model that predicts the outcome perfectly. 
The deep learning model worked the best. The random forest and logistic regression worked better than chance. The SVM model worked worse than chance. 

As the next step I am developing three other python codes to productionalize the model and maintain/update it over time according to new data. 
1. First code constantly checks the input data for new loan applications (using Google API), predicts the outcome using the best model, and emails the model prediction to the submitting personnel. The executable form of this code runs in terminal periodically (using crontab application) .
2. Another executable python code constantly checks the availability of labelled data in a google spreadsheet (using Google API), then appends the new data to the training dataset that is maintained on PostgreSQL database manager. 
3. Third code is scheduled to run every month on the updated training dataset to evaluate model performances and select the best predictive model and send the results to the relevant person to decide updating the predictive algorithm if necessary.
4. According to benefits of Random Forest model, maybe this model should be applied first to the model, since it needs minium data preparation.

