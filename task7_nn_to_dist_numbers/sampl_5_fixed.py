# -*- coding: cp1251 -*-
# CONTROLLED LEARNING
import csv 
import numpy as np
from sklearn.datasets import load_digits
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import pylab as plt
##########################################################
#function for exporting variable in CSV file
def myExport(name, input_var):
    with open (name,'w',newline = '') as csvfile:
        my_writer = csv.writer(csvfile, delimiter = ',')
        my_writer.writerows(input_var)
##########################################################        
digits = load_digits()
y = digits.target #цифра яка зображується таким масивом  digits.images - 8x8 пікселів картинка, колір [0.0 - 16.0]
n_samples = len(digits.images) # кількість цифр 1797
x = digits.images.reshape((n_samples,-1))
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0) # поділ на тренувальну та тестові виборки 
gnb = GaussianNB()
fit = gnb.fit(x_train,y_train)
predicted = fit.predict(x_test)
zipped_test = list(zip(y_test, x_test))
res_test = [ np.r_[aa] for aa in zipped_test]
res_test = [sum([[int(row[0])],list(row[1:])],[]) for row in res_test] # цифру перетворюємо на int ,  2.0->2 , ...
zipped_train = list(zip(y_train, x_train))
res_train = [ np.r_[aa] for aa in zipped_train]
res_train = [sum([[int(row[0])],list(row[1:])],[]) for row in res_train] # цифру перетворюємо на int ,  2.0->2 , ...
myExport('Data_test.csv',res_test)
myExport('Data_train.csv',res_train)
print(confusion_matrix(y_test,predicted))
images_and_prediction = list ( zip (digits.images,fit.predict(x)))
for index, ( image , prediction ) in enumerate(images_and_prediction[:36]):
    plt.subplot(6,6,index + 1)
    plt.axis('off')
    plt.imshow(image,cmap= plt.cm.gray_r,interpolation = 'nearest')
    plt.title('Pred %i' % prediction)
plt.show()    
