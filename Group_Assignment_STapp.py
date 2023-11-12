import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pickle
from math import sin, cos, pi
from statsmodels.tsa.seasonal import seasonal_decompose

# Function to load the original data
@st.cache_data
def load_original_data():
    return pd.read_pickle('bike_sharing_original.pkl')

# Function to load the preprocessed data
@st.cache_data
def load_data():
    return pd.read_pickle('bike_sharing_encoded.pkl')

# Function to load the model
#@st.cache(allow_output_mutation=True)
def load_model():
    with open('bike_prediction_lightgbm.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Encoding function
def encode_cyclic_feature(value, max_val):
    return sin(value * (2. * pi / max_val)), cos(value * (2. * pi / max_val))

# Visualization function
pio.templates.default = "seaborn"  # Set a theme for all Plotly charts

def plot_time_series_plotly(df, time_column, target_column):
    aggregated_data = df.groupby(time_column)[target_column].mean().reset_index()
    fig = px.line(aggregated_data, x=time_column, y=target_column,
                  title=f'Demand Over {time_column.capitalize()}')
    st.plotly_chart(fig)

def plot_distribution_plotly(df, column):
    fig = px.histogram(df, x=column, marginal='box', title=f'Distribution of {column.capitalize()}')
    st.plotly_chart(fig)

def plot_boxplot_plotly(df, x_column, y_column):
    fig = px.box(df, x=x_column, y=y_column, title=f'{y_column.capitalize()} Across {x_column.capitalize()}')
    st.plotly_chart(fig)

def plot_correlation_heatmap_plotly(df):
    fig = px.imshow(df.corr(), text_auto=True, title='Correlation Heatmap')
    fig.update_layout(autosize=False, width=800, height=800)
    st.plotly_chart(fig)


def plot_time_series_decomposition_plotly(df, column, freq):
    # Decompose the time series

    df['dteday'] = pd.to_datetime(df['dteday'])
    df.set_index('dteday', inplace=True)

    decomposition = seasonal_decompose(df[column], period=freq, model='additive')
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        subplot_titles=('Trend', 'Seasonal', 'Residual'))
    fig.add_trace(
        go.Scatter(x=decomposition.trend.index, y=decomposition.trend, mode='lines', name='Trend'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, mode='lines', name='Seasonal'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=decomposition.resid.index, y=decomposition.resid, mode='lines', name='Residual'),
        row=3, col=1
    )
    fig.update_layout(height=600, width=800, title_text="Time Series Decomposition")
    st.plotly_chart(fig)

# Main app function
def main():
    st.title("⭐Bike Rental Demand Prediction")
    st.markdown("---")

    # Load data and model
    original_data = load_original_data()
    data = load_data()
    model = load_model()

    # Data Visualization Section
    st.header("Data Visualizations")
    st.subheader(f"📈 Time Series Analysis")
    st.write("This trend line graph depicts bike rental demand over the course of the year. Peaks typically occur in the warmer months, suggesting a seasonal preference for biking, while dips during the colder months may reflect the impact of weather conditions on rental frequency.")
    plot_time_series_plotly(original_data, 'mnth', 'cnt')

    st.subheader(f"📉 Hourly Demand Across Weekdays")
    st.write("The box plot visualizes the distribution of bike rentals across different hours for each day of the week. Peak times likely correspond to morning and evening commutes, while the wider spread during certain hours indicates variability in user behavior.")
    plot_boxplot_plotly(original_data, 'weekday', 'cnt')

    # User selection for hour
    hour_to_filter = st.slider('🕒 Select an Hour for Analysis', min_value=0, max_value=23, value=0, step=1)
    filtered_hour_data = original_data[original_data['hr'] == hour_to_filter]

    st.subheader(f"📚 Data for Hour: {hour_to_filter}")
    st.write("Trend of bike rentals at the selected hour across different days or months.")
    plot_time_series_plotly(filtered_hour_data, 'dteday', 'cnt')

    st.write("This histogram shows the distribution of temperature levels at the selected hour")
    plot_distribution_plotly(filtered_hour_data, 'temp')

    st.subheader("🌡️ Temperature Distribution")
    st.write("This histogram shows the distribution of bike rentals across different temperature levels. A bell-shaped distribution like this often indicates that there is a 'sweet spot' range of temperatures that is most conducive to bike rentals.")
    plot_distribution_plotly(original_data, 'temp')

    # Additional Visualizations (outside the columns for better readability)

    st.subheader("🚴 Hourly Demand Distribution")
    st.write("This box plot represents the spread of bike rental demand across different hours.")
    plot_boxplot_plotly(original_data, 'hr', 'cnt')

    st.subheader("⚙️ Correlation Heatmap")
    st.write("This heatmap shows the correlations between different features in the dataset.")
    plot_correlation_heatmap_plotly(original_data)

    st.subheader("🎯 Time Series Decomposition")
    st.write("The decomposition plot breaks down the time series into Trend, Seasonal, and Residual components. The Trend line shows the overall direction of demand over time, while the Seasonal plot typically shows repeating cycles corresponding to weekly patterns, due to de high number of bike users registered per period it has this fuller solid look. The Residual plot shows the random fluctuations that are not explained by the trend or seasonality.")
    plot_time_series_decomposition_plotly(original_data, 'cnt', 168)  # Adjust 'cnt' and frequency as needed

    st.markdown("---")

    # Prediction Section
    st.header("💡 Predict Bike Rental Demand")
    hour = st.slider('🕒 Hour', 0, 23, 1)

    # Selection for month with names
    month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
                    'July', 'August', 'September', 'October', 'November', 'December']
    month = st.selectbox('📅 Month', range(1, 13), format_func=lambda x: month_names[x-1])

    # Selection for weekday with names
    weekday_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    weekday = st.selectbox('📅 Weekday', range(7), format_func=lambda x: weekday_names[x])

    # Temperature slider in Celsius (ensure these are numbers)
    temp_min_celsius = 0.0  # Example minimum temperature in Celsius
    temp_max_celsius = 40.0  # Example maximum temperature in Celsius
    temp = st.slider('🌡️ Temperature (°C)', temp_min_celsius, temp_max_celsius, step=0.5)
    temp_normalized = (temp - temp_min_celsius) / (temp_max_celsius - temp_min_celsius)

    # Humidity slider in percentage
    humidity = st.slider('💧 Humidity (%)', 0, 100, step=1)
    humidity_normalized = humidity / 100  # Normalize for the model

    windspeed = st.slider('🌬️ Windspeed', float(data['windspeed'].min()), float(data['windspeed'].max()))

    # Binary input for holiday
    holiday = st.radio('Holiday', ('Yes', 'No'))
    holiday = 1 if holiday == 'Yes' else 0

    # Automatically determine workingday
    workingday = 0 if weekday in [0, 6] or holiday == 1 else 1

    # Predict button
    if st.button('🚀Predict'):
        # Encode the inputs
        hr_sin, hr_cos = encode_cyclic_feature(hour, 24)
        mnth_sin, mnth_cos = encode_cyclic_feature(month,12)
        weekday_sin, weekday_cos = encode_cyclic_feature(weekday, 7)

        # Prepare the input DataFrame in the format your model expects
        input_data = pd.DataFrame({
            'holiday': [holiday],
            'workingday': [workingday],
            'temp': [temp_normalized],
            'hum': [humidity_normalized],
            'windspeed': [windspeed],
            'hr_sin': [hr_sin],
            'hr_cos': [hr_cos],
            'mnth_sin': [mnth_sin],
            'mnth_cos': [mnth_cos],
            'weekday_sin': [weekday_sin],
            'weekday_cos': [weekday_cos]
        })

        prediction = model.predict(input_data)
        rounded_prediction = round(prediction[0])
        st.success(f"Predicted Demand: {rounded_prediction}")


    st.markdown("---")
    st.markdown("Bike Rental Prediction App - Group 2")

if __name__ == '__main__':
    main()
