import streamlit as st
import pandas as pd
import numpy as np

st.title('Electricity Demand Forecasting')

st.info('This is a Machine Learning app that predicts based on Multiple Energy-related factors!')

df = pd.read_csv('https://raw.githubusercontent.com/Richard-Adewoye/electricity-demand-forecacsting/refs/heads/master/df_cleaned.csv')
