import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import plotly.express as px
import plotly.graph_objects as go
import pycountry


st.markdown("""
    <style>
        .glow-title {
            font-size: 48px;
            font-weight: bold;
            color: #ffffff;
            text-align: left;
            text-shadow: 
                0 0 5px #00e6e6,
                0 0 10px #00e6e6,
                0 0 20px #00e6e6,
                0 0 40px #00e6e6;
            margin-bottom: 15px;
        }

        .glow-info {
            font-size: 18px;
            font-weight: normal;
            color: #cceeff;
            text-align: center;
            background-color: rgba(0, 128, 128, 0.1);
            padding: 12px;
            border-left: 6px solid #00cccc;
            border-radius: 4px;
            box-shadow: 0 0 10px #00cccc;
            margin-bottom: 135px; /* Increased spacing below header */
        }
    </style>

    <div class="glow-title">Electricity Demand Forecasting</div>
    <div class="glow-info">
        This is a Machine Learning app that predicts based on Multiple Energy-related factors!
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    .input-indicator {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
        }
        .circle {
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background-color: lightgray;
            border: 2px solid gray;
        }
        .circle.complete {
            background-color: #28a745;
            border: 2px solid #28a745;
            position: relative;
        }
        .circle.complete::after {
            content: '\\2713';
            color: white;
            font-size: 12px;
            position: absolute;
            top: -2px;
            left: 3px;
        }
    </style>
""", unsafe_allow_html=True)



df = pd.read_csv('https://raw.githubusercontent.com/Richard-Adewoye/electricity-demand-forecacsting/refs/heads/master/df_cleaned.csv')
countries_to_drop = ['ASEAN (Ember)', 'Africa', 'Africa (EI)', 'Africa (Ember)', 'Africa (Shift)', 'Asia', 'Asia & Oceania (EIA)', 'Asia (Ember)', 'Asia Pacific (EI)', 'Asia and Oceania (Shift)', 'Australia and New Zealand (EIA)', 'CIS (EI)', 'Central & South America (EIA)', 'Central America (EI)', 'Central and South America (Shift)', 'EU28 (Shift)', 'Eastern Africa (EI)', 'Eurasia (EIA)', 'Eurasia (Shift)', 'Europe', 'Europe (EI)', 'Europe (Ember)', 'Europe (Shift)', 'European Union (27)', 'French Polynesia', 'G20 (Ember)', 'G7 (Ember)', 'High-income countries', 'IEO - Africa (EIA)', 'IEO - Middle East (EIA)','Middle East (Ember)','North America (Ember)', 'IEO OECD - Europe (EIA)', 'Low-income countries', 'Lower-middle-income countries', 'Mexico, Chile, and other OECD Americas (EIA)', 'Middle Africa (EI)', 'Middle East (EI)', 'Non-OECD (EI)', 'Non-OECD (EI)', 'Non-OPEC (EI)', 'OECD (EI)', 'OECD (EIA)', 'OECD (Ember)', 'OECD (Shift)', 'OECD - Asia And Oceania (EIA)', 'OECD - Europe (EIA)', 'OECD - North America (EIA)', 'OPEC (EI)', 'OPEC (EIA)', 'OPEC (Shift)', 'OPEC - Africa (EIA)', 'OPEC - South America (EIA)', 'Oceania', 'Oceania (Ember)', 'Other Non-OECD - America (EIA)', 'Persian Gulf (EIA)', 'Reunion', 'South and Central America (EI)', 'U.S. Territories (EIA)', 'Upper-middle-income countries', 'Western Africa (EI)', 'Western Sahara', 'World']
# Get unique country values
unique_countries = df['country'].unique()
# List of ISO-recognized countries from pycountry
valid_countries = [country.name for country in pycountry.countries]
# Flag non-standard entries
non_countries = [c for c in unique_countries if c not in valid_countries]
df = df[~df['country'].isin(countries_to_drop)]
df = df[~df['country'].isin(non_countries)]

model = joblib.load('electricity_demand_xgboost_model.pkl')

@st.cache_data
def load_data():
    return pd.read_csv('https://raw.githubusercontent.com/Richard-Adewoye/electricity-demand-forecacsting/refs/heads/master/df_cleaned.csv')

@st.cache_resource
def load_model():
    return joblib.load('electricity_demand_xgboost_model.pkl')

st.info('Map to show Average Electricity Demand')


# Compute average electricity demand per country
df = df.drop(columns=['year'])
avg_demand = df.groupby('country', as_index=False)['electricity_demand'].mean()
avg_demand.columns = ['country', 'avg_electricity_demand']

fig = px.choropleth(
    df.groupby('country', as_index=False)['electricity_demand'].mean().rename(columns={'electricity_demand': 'avg_electricity_demand'}),
    locations='country',
    locationmode='country names',
    color='avg_electricity_demand',
    color_continuous_scale='Earth',
    )

fig.update_layout(
    geo=dict(
        showframe=False,
        showcoastlines=False,
        projection_type='orthographic',
        showocean=True,
        oceancolor='lightblue',  # Set sea color here
        bgcolor='black',
        showland=True,
        landcolor='#f0e6d6',  # Elegant earth tone
        showlakes=True,
        lakecolor='lightblue',
        showcountries=True,
        countrycolor='white',
        lonaxis=dict(showgrid=True, gridcolor='lightgray'),
        lataxis=dict(showgrid=True, gridcolor='lightgray')
    ),
    margin=dict(l=0, r=0, t=50, b=0),
    height=700,  
    width=5100    
)

st.plotly_chart(fig, use_container_width=False)  # Set False to honor the manual size

with st.expander('Bar Chart'):
    st.write("### Compare Countries by Feature")

    # Exclude 'year' and non-numeric columns
    numeric_columns = df.select_dtypes(include=['float', 'int64']).columns.tolist()
    numeric_columns = [col for col in numeric_columns if col != 'year']

    selected_feature = st.selectbox("Select a feature to compare across countries", numeric_columns)

    # Group by country and compute the mean for the selected feature
    feature_by_country = df.groupby('country')[selected_feature].mean().reset_index()

    # Remove countries with duplicate values of the selected feature
    feature_by_country = feature_by_country.drop_duplicates(subset=[selected_feature])

    # Sort and select top 20
    feature_by_country = feature_by_country.sort_values(by=selected_feature, ascending=False).head(20)

    # Plot bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=feature_by_country[selected_feature],
        y=feature_by_country['country'],
        orientation='h',
        marker=dict(
            color=feature_by_country[selected_feature],
            colorscale='Viridis',
            line=dict(width=1.5, color='gray')
        ),
        opacity=0.9,
        hoverinfo='x+y'
    ))

    fig.update_layout(
        title=f'Average {selected_feature} by Country',
        xaxis_title=f'Average {selected_feature}',
        yaxis=dict(autorange="reversed"),
        margin=dict(l=0, r=0, t=50, b=0),
        height=600,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='black'
    )

    st.plotly_chart(fig, use_container_width=True)


with st.expander("Pie Chart: Country Share of Selected Feature"):
    st.write("### Pie Chart Based on Aggregated Values")

    # Let user select a numeric feature
    selected_feature = st.selectbox(
        "Select a feature for aggregation",
        df.select_dtypes(include=['float', 'int64']).columns.tolist(),
        key="bar"
    )

    # Aggregate (mean) by country
    agg_data = df.groupby('country')[selected_feature].mean().reset_index()

    # Drop countries with duplicate feature values
    agg_data = agg_data.drop_duplicates(subset=[selected_feature])

    # Sort and take top 10
    agg_data = agg_data.sort_values(by=selected_feature, ascending=False).head(10)

    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=agg_data['country'],
        values=agg_data[selected_feature],
        hole=0.3,
        pull=[0.05] * len(agg_data),
        textinfo='label+percent',
        marker=dict(line=dict(color='white', width=2))
    )])

    fig.update_layout(
        title=f"Top 10 Unique Countries by Average {selected_feature}",
        margin=dict(l=0, r=0, t=50, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)

with st.expander('World Energy Consumption Dataset'):
  st.write('**Raw Data**')
  
  #countries_to_drop = ['ASEAN (Ember)', 'Africa', 'Africa (EI)', 'Africa (Ember)', 'Africa (Shift)', 'Asia', 'Asia & Oceania (EIA)', 'Asia (Ember)', 'Asia Pacific (EI)', 'Asia and Oceania (Shift)', 'Australia and New Zealand (EIA)', 'CIS (EI)', 'Central & South America (EIA)', 'Central America (EI)', 'Central and South America (Shift)', 'EU28 (Shift)', 'Eastern Africa (EI)', 'Eurasia (EIA)', 'Eurasia (Shift)', 'Europe', 'Europe (EI)', 'Europe (Ember)', 'Europe (Shift)', 'European Union (27)', 'French Polynesia', 'G20 (Ember)', 'G7 (Ember)', 'High-income countries', 'IEO - Africa (EIA)', 'IEO - Middle East (EIA)', 'IEO OECD - Europe (EIA)', 'Low-income countries', 'Lower-middle-income countries', 'Mexico, Chile, and other OECD Americas (EIA)', 'Middle Africa (EI)', 'Middle East (EI)', 'Non-OECD (EI)', 'Non-OECD (EI)', 'Non-OPEC (EI)', 'OECD (EI)', 'OECD (EIA)', 'OECD (Ember)', 'OECD (Shift)', 'OECD - Asia And Oceania (EIA)', 'OECD - Europe (EIA)', 'OECD - North America (EIA)', 'OPEC (EI)', 'OPEC (EIA)', 'OPEC (Shift)', 'OPEC - Africa (EIA)', 'OPEC - South America (EIA)', 'Oceania', 'Oceania (Ember)', 'Other Non-OECD - America (EIA)', 'Persian Gulf (EIA)', 'Reunion', 'South and Central America (EI)', 'U.S. Territories (EIA)', 'Upper-middle-income countries', 'Western Africa (EI)', 'Western Sahara', 'World']
  #df = df[~df['country'].isin(countries_to_drop)]
  df

  mean_gdp_per_country = df.groupby('country')['gdp'].mean()

  target_encoding_map = df.groupby('country')['gdp'].mean().to_dict()
  
  st.write('**X**')
  X = df.drop(['electricity_demand'], axis=1)
  st.write(X)
  
  st.write('**y**')
  y = df.electricity_demand
  st.write(y)

# Indicator tracking dictionary
if 'completed_inputs' not in st.session_state:
    st.session_state.completed_inputs = {}

def input_with_indicator(label, min_val, max_val, default):
    key = f"slider_{label}"
    val = st.slider(label, min_val, max_val, default, key=key)
    is_complete = val != default
    st.session_state.completed_inputs[key] = is_complete
    circle_class = "circle complete" if is_complete else "circle"
    st.markdown(f"<div class='input-indicator'><div class='{circle_class}'></div><label>{label}</label></div>", unsafe_allow_html=True)
    return val

with st.sidebar:
    st.header('Please input the required features')

    countries = sorted(target_encoding_map.keys())
    search_term = st.text_input("Search for a country")
    filtered_countries = [c for c in countries if search_term.lower() in c.lower()]

    if filtered_countries:
        default_index = filtered_countries.index("Nigeria") if "Nigeria" in filtered_countries else 0
        country = st.selectbox('Select a Country', filtered_countries, index=default_index)
    else:
        st.warning("No Country matches your search")
        country = None

    if country:
        country_encoded_value = mean_gdp_per_country[country]

        year = input_with_indicator('year', 2023, 2100, 2025)
        population = input_with_indicator('population', 1000000000, 6000000000, 3000000000)
        gdp = input_with_indicator('gdp', 134586329843, 912328463859, 123456789)
        coal_prod_change_pct = input_with_indicator('coal_prod_change_pct', 37.43, 90.32, 65.43)
        coal_prod_change_twh = input_with_indicator('coal_prod_change_twh', 37.43, 90.32, 65.43)
        coal_prod_per_capita = input_with_indicator('coal_prod_per_capita', 37.43, 90.32, 65.43)
        coal_production = input_with_indicator('coal_production', 37.43, 90.32, 65.43)
        electricity_generation = input_with_indicator('electricity_generation', 37.43, 90.32, 65.43)
        energy_cons_change_pct = input_with_indicator('energy_cons_change_pct', 37.43, 90.32, 65.43)
        energy_cons_change_twh = input_with_indicator('energy_cons_change_twh', 37.43, 90.32, 65.43)
        energy_per_capita = input_with_indicator('energy_per_capita', 37.43, 90.32, 65.43)
        energy_per_gdp = input_with_indicator('energy_per_gdp', 37.43, 90.32, 65.43)
        gas_prod_change_pct = input_with_indicator('gas_prod_change_pct', 37.43, 90.32, 65.43)
        gas_prod_change_twh = input_with_indicator('gas_prod_change_twh', 37.43, 90.32, 65.43)
        gas_prod_per_capita = input_with_indicator('gas_prod_per_capita', 37.43, 90.32, 65.43)
        gas_production = input_with_indicator('gas_production', 37.43, 90.32, 65.43)
        hydro_electricity = input_with_indicator('hydro_electricity', 37.43, 90.32, 65.43)
        hydro_share_elec = input_with_indicator('hydro_share_elec', 37.43, 90.32, 65.43)
        low_carbon_elec_per_capita = input_with_indicator('low_carbon_elec_per_capita', 37.43, 90.32, 65.43)
        low_carbon_electricity = input_with_indicator('low_carbon_electricity', 37.43, 90.32, 65.43)
        low_carbon_share_elec = input_with_indicator('low_carbon_share_elec', 37.43, 90.32, 65.43)
        nuclear_elec_per_capita = input_with_indicator('nuclear_elec_per_capita', 37.43, 90.32, 65.43)
        nuclear_electricity = input_with_indicator('nuclear_electricity', 37.43, 90.32, 65.43)
        nuclear_share_elec = input_with_indicator('nuclear_share_elec', 37.43, 90.32, 65.43)
        oil_prod_change_pct = input_with_indicator('oil_prod_change_pct', 37.43, 90.32, 65.43)
        oil_prod_change_twh = input_with_indicator('oil_prod_change_twh', 37.43, 90.32, 65.43)
        oil_prod_per_capita = input_with_indicator('oil_prod_per_capita', 37.43, 90.32, 65.43)
        oil_production = input_with_indicator('oil_production', 37.43, 90.32, 65.43)
        other_renewable_electricity = input_with_indicator('other_renewable_electricity', 37.43, 90.32, 65.43)
        other_renewables_elec_per_capita = input_with_indicator('other_renewables_elec_per_capita', 37.43, 90.32, 65.43)
        other_renewables_share_elec = input_with_indicator('other_renewables_share_elec', 37.43, 90.32, 65.43)
        primary_energy_consumption = input_with_indicator('primary_energy_consumption', 37.43, 90.32, 65.43)
        renewables_elec_per_capita = input_with_indicator('renewables_elec_per_capita', 37.43, 90.32, 65.43)
        renewables_electricity = input_with_indicator('renewables_electricity', 37.43, 90.32, 65.43)
        renewables_share_elec = input_with_indicator('renewables_share_elec', 37.43, 90.32, 65.43)
        solar_elec_per_capita = input_with_indicator('solar_elec_per_capita', 37.43, 90.32, 65.43)
        solar_electricity = input_with_indicator('solar_electricity', 37.43, 90.32, 65.43)
        solar_share_elec = input_with_indicator('solar_share_elec', 37.43, 90.32, 65.43)
        wind_elec_per_capita = input_with_indicator('wind_elec_per_capita', 37.43, 90.32, 65.43)
        wind_electricity = input_with_indicator('wind_electricity', 37.43, 90.32, 65.43)
        wind_share_elec = input_with_indicator('wind_share_elec', 37.43, 90.32, 65.43)


        if st.button("Predict"):
            input_dict = {
                'country_encoded': country_encoded_value,
                'year': year,
                'population': population,
                'gdp': gdp,
                'coal_prod_change_pct': coal_prod_change_pct,
                'coal_prod_change_twh': coal_prod_change_twh,
                'coal_prod_per_capita': coal_prod_per_capita,
                'coal_production': coal_production,
                'electricity_generation': electricity_generation,
                'energy_cons_change_pct': energy_cons_change_pct,
                'energy_cons_change_twh': energy_cons_change_twh,
                'energy_per_capita': energy_per_capita,
                'energy_per_gdp': energy_per_gdp,
                'gas_prod_change_pct': gas_prod_change_pct,
                'gas_prod_change_twh': gas_prod_change_twh,
                'gas_prod_per_capita': gas_prod_per_capita,
                'gas_production': gas_production,
                'hydro_electricity': hydro_electricity,
                'hydro_share_elec': hydro_share_elec,
                'low_carbon_elec_per_capita': low_carbon_elec_per_capita,
                'low_carbon_electricity': low_carbon_electricity,
                'low_carbon_share_elec': low_carbon_share_elec,
                'nuclear_elec_per_capita': nuclear_elec_per_capita,
                'nuclear_electricity': nuclear_electricity,
                'nuclear_share_elec': nuclear_share_elec,
                'oil_prod_change_pct': oil_prod_change_pct,
                'oil_prod_change_twh': oil_prod_change_twh,
                'oil_prod_per_capita': oil_prod_per_capita,
                'oil_production': oil_production,
                'other_renewable_electricity': other_renewable_electricity,
                'other_renewables_elec_per_capita': other_renewables_elec_per_capita,
                'other_renewables_share_elec': other_renewables_share_elec,
                'primary_energy_consumption': primary_energy_consumption,
                'renewables_elec_per_capita': renewables_elec_per_capita,
                'renewables_electricity': renewables_electricity,
                'renewables_share_elec': renewables_share_elec,
                'solar_elec_per_capita': solar_elec_per_capita,
                'solar_electricity': solar_electricity,
                'solar_share_elec': solar_share_elec,
                'wind_elec_per_capita': wind_elec_per_capita,
                'wind_electricity': wind_electricity,
                'wind_share_elec': wind_share_elec,
            }

            input_df = pd.DataFrame([input_dict])
            expected_features = model.get_booster().feature_names

            for feature in expected_features:
                if feature not in input_df.columns:
                    input_df[feature] = 0
            input_df = input_df[expected_features]

            prediction = model.predict(input_df)

            # Styling fix
            st.markdown("""
                <style>
                    .main .block-container {
                        padding-left: 0rem;
                        padding-right: 0rem;
                    }
                </style>
            """, unsafe_allow_html=True)

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
            """, unsafe_allow_html=True)

