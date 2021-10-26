import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from PIL import Image

st.title("Towards Data Science")
st.write("""# A SIMPLE DATA APP WITH STREAMLIT BY ME""")
st.write("""### LETS EXPLORE THE ML WORLD""")
         

def main():
    activities=["EDA","Visualization","model","About me"]
    option=st.sidebar.selectbox("SELECTION OPTION:",activities)
    if option=="EDA":
        st.subheader("Exploaratory Data Analysis")
        data=st.file_uploader("Upload dataset:",type=["csv","xlsx","txt","json"])
        st.success("DATA SUCCESSFULLY LOADED")
        if data is not None:
            df=pd.read_csv(data)
            st.dataframe(df.head(10))
            if st.checkbox("display shape"):
                st.write(df.shape)
            if st.checkbox("display columns"):
                st.write(df.columns)
            if st.checkbox("Select multi columns"):
                selected_columns=st.multiselect("SELECT PREFERED COLUMNS",df.columns)
                df1=df[selected_columns]
                st.dataframe(df1)
            if st.checkbox("Display Summary"):
                st.write(df.describe().T)
            if st.checkbox("Display Null Vlues"):
                st.write(df.isnull().sum())
            if st.checkbox("Display Correlation of data columns"):
                st.write(df.corr())
    
    
    
    elif option=="Visualization":
        st.subheader("Visualize your data")
        data=st.file_uploader("Upload dataset:",type=["csv","xlsx","txt","json"])
        st.success("Data Uploaded Successfully..")
        if data is not None:
            df=pd.read_csv(data)
            st.dataframe(df.head(10))
            
            if st.checkbox("Select MUltiple Columns to Ploat"):
                selected_columns=st.multiselect("Select your preferes columns",df.columns)
                df1=df[selected_columns]
                st.dataframe(df1)
            
            if st.checkbox("Display Heatmap"):
                st.write(sns.heatmap(df1.corr()),vmax=2,square=True,annot=True,cmap="viridis")
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()
            if st.checkbox("Display pairplot"):
                st.write(sns.pairplot(df,diag_kind="kde"))
                st.pyplot()
            if st.checkbox("Display Pie chart"):
                all_columns=df.columns.to_list()
                pie_columns=st.selectbox("Select Column to Display",all_columns)
                piechart=df[pie_columns].value_counts().plot.pie(autopct="%1.1f%%")
                st.write(piechart)
                st.pyplot()
        
    elif option=="model":
        st.subheader("Model Building")
        data=st.file_uploader("Upload dataset:",type=["csv","xlsx","txt","json"])
        st.success("Data Uploaded Successfully..")
        if data is not None:
            df=pd.read_csv(data)
            st.dataframe(df.head(10))
            
            if st.checkbox("Select Multiple Columns"):
                new_data=st.multiselect("Select Your Prefered Columns",df.columns)
                df1=df[new_data]
                st.dataframe(df1)
                
                #dividing into x and y
                X=df1.iloc[:,0:-1]
                y=df1.iloc[:,-1]
                
            RANDOM_STATE=st.sidebar.slider("RANDOM_STATE",1,200)
            
            classifier_name=st.sidebar.selectbox('SELECT YOUR PREFERED CLASSIFIER',('KNN','SVM',"LR",'NAIVE_BAYES',"decission trees"))
             
            
            def add_parameter(name_of_clf):
                params=dict()
                if name_of_clf=="SVM":
                    C=st.sidebar.slider("C",0.01,20.00)
                    params['C']=C
                else:
                    name_of_clf=="KNN"
                    K=st.sidebar.slider('K',1,20)
                    params['K']=K
                return params
                    
            params=add_parameter(classifier_name)

                #defining function our classifier
            def get_classifier(name_of_clf,params):
                clf=None
                if name_of_clf=="SVM":
                    clf=SVC(C=params["C"])
                elif name_of_clf=="KNN":
                    clf=KNeighborsClassifier(n_neighbors=params["K"])
                elif name_of_clf=="LR":
                    clf=LogisticRegression()
                elif name_of_clf=="NAIVE_BAYES":
                    clf=GaussianNB()
                elif name_of_clf=="decission trees":
                    clf=DecisionTreeClassifier()
                else:
                    st.warning("Select Your Choice Of Algorithm")
                return clf
            clf=get_classifier(classifier_name,params)
                
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=RANDOM_STATE)
            clf.fit(X_train,y_train)
            y_pred=clf.predict(X_test)
            st.write("predictions: ",y_pred)
            accuracy=accuracy_score(y_test,y_pred)
            st.write("classifier name is : ",classifier_name)
            st.write("accuracy score for your model is ",accuracy)
                     
                
                
                
                
if __name__=="__main__":
    main()