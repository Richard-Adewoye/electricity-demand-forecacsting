import streamlit as st
import pandas as pd
import numpy as np

st.title('Electricity Demand Forecasting')

st.info('This is a Machine Learning app that predicts based on Multiple Energy-related factors!')

with st.expander('World Energy Consumption Dataset'):
  st.write('**Raw Data**')
  df = pd.read_csv('https://raw.githubusercontent.com/Richard-Adewoye/electricity-demand-forecacsting/refs/heads/master/df_cleaned.csv')
  df

  st.write('**X**')
  X = df.drop(['electricity_demand'], axis=1)
  st.write(X)
  
  st.write('**y**')
  y = df.electricity_demand
  st.write(y)
  
with st.expander('Data visualisation'):
  # Let user pick a numeric feature
  numeric_columns = df.select_dtypes(include=['float', 'int64']).columns.tolist()
  selected_feature = st.selectbox("Select a feature to plot", numeric_columns)

  # Plot scatter chart for each country
  st.write(f"Scatter plot of `{selected feature}` across countries")
  st.scatter_chart(df[['country', selected_feature]])
