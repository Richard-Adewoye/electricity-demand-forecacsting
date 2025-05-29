import streamlit as st
import pandas as pd
import numpy as np

st.title('Electricity Demand Forecasting')

st.info('This is a Machine Learning app that predicts based on Multiple Energy-related factors!')

with st.expander('World Energy Consumption Dataset'):
  st.write('**Raw Data**')
  df = pd.read_csv('https://raw.githubusercontent.com/Richard-Adewoye/electricity-demand-forecacsting/refs/heads/master/df_cleaned.csv')
  df
