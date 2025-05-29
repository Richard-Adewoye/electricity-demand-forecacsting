import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib

st.title('Electricity Demand Forecasting')

st.info('This is a Machine Learning app that predicts based on Multiple Energy-related factors!')

model = joblib.load('electricity_demand_xgboost_model.pkl')

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

  # Create scatter plot using Altair
  chart = alt.Chart(df).mark_circle(size=60).encode(
    x=alt.X('country:N', sort='-y'), # x-axis as categorical
    y=alt.Y(selected_feature, title=selected_feature),
    tooltip=['country', selected_feature]
  ).properties(
    width=700,
    height=400)

  st.altair_chart(chart, use_container_width=True)

# Data input
with st.sidebar:
  st.header('Please input the required features')
  #""year, population, gdp, coal_prod_change_pct, coal_prod_change_twh, coal_prod_per_capita, coal_production, electricity_demand, electricity_generation, energy_cons_change_pct, energy_cons_change_twh, energy_per_capita, energy_per_gdp, gas_prod_change_pct, gas_prod_change_twh, gas_prod_per_capita, gas_production, hydro_electricity, hydro_share_elec, low_carbon_elec_per_capita, low_carbon_electricity, low_carbon_share_elec, nuclear_elec_per_capita, nuclear_electricity, nuclear_share_elec, oil_prod_change_pct, oil_prod_change_twh, oil_prod_per_capita, oil_production, other_renewable_electricity, other_renewables_elec_per_capita, other_renewables_share_elec, primary_energy_consumption, renewables_elec_per_capita, renewables_electricity, renewables_share_elec, solar_elec_per_capita, solar_electricity, solar_share_elec, wind_elec_per_capita, wind_electricity, wind_share_elec""
  # To get the list of unique countries
  countries = sorted(df['country'].unique())
  
  country = st.selectbox('select a country:', countries)
  year = st.slider('year', 2023, 2400, 2025)
  population = st.slider('population', 1000000000, 6000000000, 3000000000)
  gdp = st.slider('gdp', 134586329843, 912328463859, 123456789)
  coal_prod_change_pct = st.slider('coal_prod_change_pct', 37.43, 90.32, 65.43)
  coal_prod_change_twh = st.slider('coal_prod_change_twh', 37.43, 90.32, 65.43)
  coal_prod_per_capita = st.slider('coal_prod_per_capita', 37.43, 90.32, 65.43)
  coal_production = st.slider('coal_production', 37.43, 90.32, 65.43)
  electricity_generation = st.slider('electricity_generation', 37.43, 90.32, 65.43)
  energy_cons_change_pct = st.slider('energy_cons_change_pct', 37.43, 90.32, 65.43)
  energy_cons_change_twh = st.slider('energy_cons_change_twh', 37.43, 90.32, 65.43)
  energy_per_capita = st.slider('energy_per_capita', 37.43, 90.32, 65.43)
  energy_per_gdp = st.slider('energy_per_gdp', 37.43, 90.32, 65.43)
  gas_prod_change_pct = st.slider('gas_prod_change_pct', 37.43, 90.32, 65.43)
  gas_prod_change_twh = st.slider('gas_prod_change_twh', 37.43, 90.32, 65.43)
  gas_prod_per_capita = st.slider('gas_prod_per_capita', 37.43, 90.32, 65.43)
  gas_production = st.slider('gas_production', 37.43, 90.32, 65.43)
  hydro_electricity = st.slider('hydro_electricity', 37.43, 90.32, 65.43)
  hydro_share_elec = st.slider('hydro_share_elec', 37.43, 90.32, 65.43)
  low_carbon_elec_per_capita = st.slider('low_carbon_elec_per_capita', 37.43, 90.32, 65.43)
  low_carbon_electricity = st.slider('low_carbon_electricity', 37.43, 90.32, 65.43)
  low_carbon_share_elec = st.slider('low_carbon_share_elec', 37.43, 90.32, 65.43)
  nuclear_elec_per_capita = st.slider('nuclear_elec_per_capita', 37.43, 90.32, 65.43)
  nuclear_electricity = st.slider('nuclear_electricity', 37.43, 90.32, 65.43)
  nuclear_share_elec = st.slider('nuclear_share_elec', 37.43, 90.32, 65.43)
  oil_prod_change_pct = st.slider('oil_prod_change_pct', 37.43, 90.32, 65.43)
  oil_prod_change_twh = st.slider('oil_prod_change_twh', 37.43, 90.32, 65.43)
  oil_prod_per_capita = st.slider('oil_prod_per_capita', 37.43, 90.32, 65.43)
  oil_production = st.slider('oil_production', 37.43, 90.32, 65.43)
  other_renewable_electricity = st.slider('other_renewable_electricity', 37.43, 90.32, 65.43)
  other_renewables_elec_per_capita = st.slider('other_renewables_elec_per_capita', 37.43, 90.32, 65.43)
  other_renewables_share_elec = st.slider('other_renewables_share_elec', 37.43, 90.32, 65.43)
  primary_energy_consumption = st.slider('primary_energy_consumption', 37.43, 90.32, 65.43)
  renewables_elec_per_capita = st.slider('renewables_elec_per_capita', 37.43, 90.32, 65.43)
  renewables_electricity = st.slider('renewables_electricity', 37.43, 90.32, 65.43)
  renewables_share_elec = st.slider('renewables_share_elec', 37.43, 90.32, 65.43)
  solar_elec_per_capita = st.slider('solar_elec_per_capita', 37.43, 90.32, 65.43)
  solar_electricity = st.slider('solar_electricity', 37.43, 90.32, 65.43)
  solar_share_elec = st.slider('solar_share_elec', 37.43, 90.32, 65.43)
  wind_elec_per_capita = st.slider('wind_elec_per_capita', 37.43, 90.32, 65.43)
  wind_electricity = st.slider('wind_electricity', 37.43, 90.32, 65.43)
  wind_share_elec = st.slider('wind_share_elec', 37.43, 90.32, 65.43)

  if st.button("Predict"):
    input_dict = {
      'country':country,
      'year':year,
      'population':population,
      'gdp':gdp,
      'coal_prod_change_pct':coal_prod_change_pct,
      'coal_prod_change_twh':coal_prod_change_twh,
      'coal_prod_per_capita':coal_prod_per_capita,
      'coal_production':coal_production,
      'electricity_demand':electricity_demand,
      'electricity_generation':electricity_generation,
      'energy_cons_change_pct':energy_cons_change_pct,
      'energy_cons_change_twh':energy_cons_change_twh,
      'energy_per_capita':energy_per_capita,
      'energy_per_gdp':energy_per_gdp,
      'gas_prod_change_pct':gas_prod_change_pct,
      'gas_prod_change_twh':gas_prod_change_twh,
      'gas_prod_per_capita':gas_prod_per_capita,
      'gas_production':gas_production,
      'hydro_electricity':hydro_electricity,
      'hydro_share_elec':hydro_share_elec,
      'low_carbon_elec_per_capita':low_carbon_elec_per_capita,
      'low_carbon_electricity':low_carbon_electricity,
      'low_carbon_share_elec':low_carbon_share_elec,
      'nuclear_elec_per_capita':nuclear_elec_per_capita,
      'nuclear_electricity':nuclear_electricity,
      'nuclear_share_elec':nuclear_share_elec,
      'oil_prod_change_pct':oil_prod_change_pct,
      'oil_prod_change_twh':oil_prod_change_twh,
      'oil_prod_per_capita':oil_prod_per_capita,
      'oil_production':oil_production,
      'other_renewable_electricity':other_renewable_electricity,
      'other_renewables_elec_per_capita':other_renewables_elec_per_capita,
      'other_renewables_share_elec':other_renewables_share_elec,
      'primary_energy_consumption':primary_energy_consumption,
      'renewables_elec_per_capita':renewables_elec_per_capita,
      'renewables_electricity':renewables_electricity,
      'renewables_share_elec':renewables_share_elec,
      'solar_elec_per_capita':solar_elec_per_capita,
      'solar_electricity':solar_electricity,
      'solar_share_elec':solar_share_elec,
      'wind_elec_per_capita':wind_elec_per_capita,
      'wind_electricity':wind_electricity,
      'wind_share_elec':wind_share_elec,
    }

  # Covert to Dataframe
  input_df = pd.DataFrame([input_dict])

  # Make prediction
  prediction = model.predict(input_df)

  st.success(f"Predicted Electricity Demand for {country} in {year} is: {prediction[0]:,.2f}")
