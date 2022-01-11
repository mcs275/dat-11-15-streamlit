# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle

st.title("Predicting Hourly Bikeshare Rentals")

url = r"https://github.com/mcs275/dat-11-15-streamlit/blob/main/bikeshare.csv"

num_rows = st.sidebar.number_input('Select Number of Rows to Load', 
                                   min_value = 1000, 
                                   max_value = 50000, 
                                   step = 1000)

section = st.sidebar.radio('Choose Application Section', ['Data Explorer', 
                                                          'Model Explorer'])

@st.cache
def load_data(num_rows):
    df = pd.read_csv(url, nrows = num_rows)
    return df

@st.cache
def create_grouping(x_axis, y_axis):
    grouping = df.groupby(x_axis)[y_axis].mean()
    return grouping

def load_model():
    with open('pipe.pkl', 'rb') as pickled_mod:
        model = pickle.load(pickled_mod)
    return model

df = load_data(num_rows)

if section == 'Data Explorer':
    
    
    x_axis = st.sidebar.selectbox("Choose column for X-axis", 
                                  df.select_dtypes(include = np.object).columns.tolist())
    
    y_axis = st.sidebar.selectbox("Choose column for y-axis", ['count'])
    
    chart_type = st.sidebar.selectbox("Choose Your Chart Type", 
                                      ['line', 'bar', 'area'])
    
    if chart_type == 'line':
        grouping = create_grouping(x_axis, y_axis)
        st.line_chart(grouping)
        
    elif chart_type == 'bar':
        grouping = create_grouping(x_axis, y_axis)
        st.bar_chart(grouping)
    elif chart_type == 'area':
        fig = px.strip(df[[x_axis, y_axis]], x=x_axis, y=y_axis)
        st.plotly_chart(fig)
    
    st.write(df)
    
else:
    st.text("Choose Options to the Side to Explore the Model")
    model = load_model()
    
    datetime_val = st.sidebar.selectbox("Choose Date and Time", 
                                  df['datetime'].unique().tolist())
    last_hour = st.sidebar.number_input("How many bike shares rented prior hour?", min_value = 0,
                                        max_value = 100, step = 1, value = 20)
    season = st.sidebar.selectbox("Season", 
                                       df['season'].unique().tolist())
    
    sample = {
    'datetime': datetime_val,
    '1hourago': last_hour,
    'season': season
    }

    sample = pd.DataFrame(sample, index = [0])
    prediction = model.predict(sample)[0]
    
    st.title(f"Predicted Bike Shares: {int(prediction)}")
    