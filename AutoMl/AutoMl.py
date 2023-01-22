from operator import index
import streamlit as st
import plotly.express as px
from pycaret.regression import *
import pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os 
import sweetviz as sv

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar: 
    st.title("AUTOML APP BY VARUN PADHA")
    choice = st.radio("Options", ["Home","Upload","Sweetviz EDA","Pandas Profiling","Modelling", "Download"])
   
if choice == "Home":
 st.title("AutoML App BY Varun Padha")
 st.info("App that helps you create the best machine learning model for your data for free.Makes the eda easy and powerful by using sweetwiz and Pandas Profiling and download the best model in a pickel file")
 st.image("https://imgs.search.brave.com/33AflSaJjRmdYV0wRkGuYXXngGCjBFU3nGQLedPj06A/rs:fit:1200:1200:1/g:ce/aHR0cHM6Ly9pMC53/cC5jb20vYmFuZS10/ZWNoLmNvbS93cC1j/b250ZW50L3VwbG9h/ZHMvMjAxNS8xMC9W/LnBuZz9zc2w9MQ",width=300)
 st.markdown("![Alt Text](https://media.giphy.com/media/l3vR7850ncoUC9BdK/giphy.gif)")
 

 
if choice == "Upload":
    st.title("Upload Your Dataset")
    st.markdown("![Alt Text](https://media.giphy.com/media/UEGwYCVTBFa9tJEf66/giphy.gif)")
    file = st.file_uploader("Please Upload Your Dataset or CSV file")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "Pandas Profiling": 
    st.snow()
    st.title("Data Analysis Using Pandas Profiling")
    profile_df = df.profile_report()
    st_profile_report(profile_df)
if choice == "Sweetviz EDA":
    advert_report = sv.analyze(df)
    advert_report.show_html('SweetViz.html')


if choice == "Modelling": 
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    st.markdown("![Alt Text](https://media.giphy.com/media/HUplkVCPY7jTW/giphy.gif)") 
    if st.button('Run Modelling'):
        st.balloons()
        setup(df, target=chosen_target, silent=True)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')

if choice == "Download": 
    st.markdown("![Alt Text](https://media.giphy.com/media/XaLnoepP2IwFnUXdvb/giphy.gif)") 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")
        st.success('Thanks for using my AutoMl app', icon="😊")
        st.success("Star the Github peoject if you like",icon="❤️")
