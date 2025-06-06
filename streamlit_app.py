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
  countries_to_drop = ['ASEAN (Ember)', 'Africa', 'Africa (EI)', 'Africa (Ember)', 'Africa (Shift)', 'Asia', 'Asia & Oceania (EIA)', 'Asia (Ember)', 'Asia Pacific (EI)', 'Asia and Oceania (Shift)', 'Australia and New Zealand (EIA)', 'CIS (EI)', 'Central & South America (EIA)', 'Central America (EI)', 'Central and South America (Shift)', 'EU28 (Shift)', 'Eastern Africa (EI)', 'Eurasia (EIA)', 'Eurasia (Shift)', 'Europe', 'Europe (EI)', 'Europe (Ember)', 'Europe (Shift)', 'European Union (27)', 'French Polynesia', 'G20 (Ember)', 'G7 (Ember)', 'High-income countries', 'IEO - Africa (EIA)', 'IEO - Middle East (EIA)', 'IEO OECD - Europe (EIA)', 'Low-income countries', 'Lower-middle-income countries', 'Mexico, Chile, and other OECD Americas (EIA)', 'Middle Africa (EI)', 'Middle East (EI)', 'Non-OECD (EI)', 'Non-OECD (EI)', 'Non-OPEC (EI)', 'OECD (EI)', 'OECD (EIA)', 'OECD (Ember)', 'OECD (Shift)', 'OECD - Asia And Oceania (EIA)', 'OECD - Europe (EIA)', 'OECD - North America (EIA)', 'OPEC (EI)', 'OPEC (EIA)', 'OPEC (Shift)', 'OPEC - Africa (EIA)', 'OPEC - South America (EIA)', 'Oceania', 'Oceania (Ember)', 'Other Non-OECD - America (EIA)', 'Persian Gulf (EIA)', 'Reunion', 'South and Central America (EI)', 'U.S. Territories (EIA)', 'Upper-middle-income countries', 'Western Africa (EI)', 'Western Sahara', 'World']
  df = df[~df['country'].isin(countries_to_drop)]
  df

  mean_gdp_per_country = df.groupby('country')['gdp'].mean()

  target_encoding_map = df.groupby('country')['gdp'].mean().to_dict()
  
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
  countries = sorted(target_encoding_map.keys())

  search_term = st.text_input("Search for a country")
  filtered_countries = [c for c in df['country'].unique() if search_term.lower() in c.lower()]
  country = st.selectbox('Select a Country', sorted(filtered_countries)) if filtered_countries else st.warning("No Country matches your search")

  # Map selected country to encoded value
  country_encoded_value = mean_gdp_per_country[country]
  
   
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
      'country_encoded':country_encoded_value,
      'year':year,
      'population':population,
      'gdp':gdp,
      'coal_prod_change_pct':coal_prod_change_pct,
      'coal_prod_change_twh':coal_prod_change_twh,
      'coal_prod_per_capita':coal_prod_per_capita,
      'coal_production':coal_production,
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

    # Get expected features
    expected_features = model.get_booster().feature_names


    # Ensure all expected features are present in input_df
    for feature in expected_features:
      if feature not in input_df.columns:
        input_df[feature] = 0 # or another default

    # keep only the expected columns, in order
    input_df = input_df[expected_features]

    # Make prediction
    prediction = model.predict(input_df)

    st.markdown(f"""
      <div style='
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #d4edda;
        color: #155724;
        padding: 40px 20px;
        font-size: 28px;
        font-weight: bold;
        text-align: center;
        border-top: 5px solid #28a745;
        z-index: 9999;
        '>
        Predicted Electricity Demand for {country} in {year} is: {prediction[0]:,.2f}
        </div>
        """, unsafe_allow_html=True
               )
