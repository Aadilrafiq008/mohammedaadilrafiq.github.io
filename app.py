import streamlit as st 
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

st.title('helo')

dataset_select = st.sidebar.selectbox("select dataset ",('iris', 'breast cancer', 'wine dataset'))
classifier = st.sidebar.selectbox("select classifier",("KNN", "SVM","RANDOM FOREST"))

def dataset_fun(dataset_select):
    if dataset_select == 'iris':
        data = datasets.load_iris()

    elif dataset_select == 'breast cancer':
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X,y

X, y = dataset_fun(dataset_select)
st.write('shape of datasets', X.shape)
st.write('no. of classes', len(np.unique(y)))

def add_parametes(clf_name):
    param = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K",1,15)
        param['K'] = K
        
    elif clf_name == "SVM":
        C = st.sidebar.slider("C",0.1,10.99)
        param['C'] = C
    else:
        max_depth = st.sidebar.slider("max_depth", 2,15)
        n_estimator = st.sidebar.slider("n_estimator", 1,100)
        param['max_depth'] = max_depth
        param['n_estimator'] = n_estimator
    return param

param = add_parametes(classifier)

def get_classfier(classifier, param):
    if classifier == "KNN":
        clf = KNeighborsClassifier(n_neighbors=param['K'])           
    elif classifier == "SVM":
        clf = SVC(C=param['C'])
    else:
        clf= RandomForestClassifier(n_estimators=param['n_estimator'],max_depth=param['max_depth'])
    return clf
clf = get_classfier(classifier, param)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

clf.fit(X_train,y_train)
pre = clf.predict(X_test)
# acc = accuracy_score(X_test,y_test)
acc = accuracy_score(y_test, pre)

st.write(f"Classifier is = {classifier}")
st.write(f"accuracy_score is = {acc}")


pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2,
        c=y, alpha=0.8,
        cmap='viridis')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

#plt.show()
st.pyplot(fig)