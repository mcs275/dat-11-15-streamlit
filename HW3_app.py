# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle

st.title("Predicting Number of Bikes Rented Per Hour for City Bikeshare Program")

url = r"https://raw.githubusercontent.com/mcs275/dat-class-repo/main/Homework/Unit2/data/bikeshare.csv"

num_rows = st.sidebar.number_input('Select Number of Rows to Load', 
                                   min_value = 100, 
                                   max_value = 11000, 
                                   step = 100)

section = st.sidebar.radio('Choose Application Section', ['Data Explorer', 
                                                          'Model Explorer'])

@st.cache
def load_data(num_rows):
    df = pd.read_csv(url, nrows = num_rows)
    df = df.sort_values(by=['datetime'])
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
       # grouping = create_grouping(x_axis, y_axis)
        group = df.groupby(x_axis)[y_axis].mean()
        fig = px.line(group)
     ##   fig.update_layout(autosize=False, width=1000, height=500)
        st.plotly_chart(fig)
        
    elif chart_type == 'bar':
       ## grouping = create_grouping(x_axis, y_axis)
       group = df.groupby(x_axis)[y_axis].mean()
       fig = px.bar(group)
       fig.update_layout()
       st.plotly_chart(fig)
        
    elif chart_type == 'area':
        fig = px.strip(df[[x_axis, y_axis]], x=x_axis, y=y_axis)
        st.plotly_chart(fig)
    
    st.write(df)
    
else:
    st.text("Choose Options to the Side to Explore the Model")
    model = load_model()
    
    
    days_val = st.sidebar.number_input("Enter Number of Days since start of data", 
                                min_value = 0, max_value = 10000, step = 1, value = 20)
    
    time_of_day = st.sidebar.number_input("Choose Time of Day  (1=Early am, 2=AM rush, 3=midday, 4=PM rush, 5=Night",
                                          min_value=1, max_value=5, value = 2)
                                         
                              
    
    dayofweek_val = st.sidebar.number_input("Choose Day of Week (0=Monday, 6=Sunday)", 
                                            min_value=0, max_value =6, value=0)

    weather_val = st.sidebar.number_input("Choose Weather (1=clear sies, 2=partly cloudly, 3=light storms, 4=heavy storms", 
                                  min_value=1, max_value =4, value=3)
   
    
    season = st.sidebar.number_input("Choose Season (1=spring, 2=summer, 3=fall, 4=winter", 
                                       min_value=1, max_value =4, value=3)
    
    holiday = st.sidebar.selectbox("Select if it's a holiday (1=Yes, 0=No)", 
                                       df['holiday'].unique().tolist())
    
    workingday = st.sidebar.selectbox("Select if a working day (1=Yes, 0=No)", 
                                       df['workingday'].unique().tolist())
    
    temp = st.sidebar.number_input("What is the temperature (celsius)?", min_value = 0.0,
                                    max_value = 50.0, step = 0.5, value = 20.0)
    
    avg_temp = st.sidebar.number_input("What is the average day temperature?", min_value = 0.0,
                                    max_value = 50.0, step = 0.5, value = 23.0)
    
    humidity = st.sidebar.number_input("What is the humidity?", min_value = 0,
                                    max_value = 100, step = 5, value = 25)
    
    windspeed = st.sidebar.number_input("What is the wind speed?", min_value = 0,
                                    max_value = 60, step = 5, value = 10)  
    
    last_hour = st.sidebar.number_input("How many bike rented prior hour?", min_value = 0,
                                    max_value = 1500, step = 1, value = 191)
    
    yesterday = st.sidebar.number_input("How many bikes rented 24 hours ago?", min_value = 0,args=(), 
                                         max_value = 1500, step = 1, value = 191)
      
    week_ago = st.sidebar.number_input("How many bikes rented 7 days ago", min_value = 0,
                                    max_value = 1500, step = 1, value = 191)
    
    month_ago = st.sidebar.number_input("How many bikes rented 1 month ago?", min_value = 0,
                                    max_value = 1500, step = 1, value = 191)
    sample = {
        'Days': days_val,
        'time_of_day': time_of_day,
        'day_of_week': dayofweek_val,
        'weather': weather_val,
        '1hourago': last_hour,
        'season': season,
        'holiday': holiday,
        'workingday': workingday,
        'temp': temp,
        'atemp': avg_temp,
        'humidity': humidity,
        'windspeed': windspeed,
        '24hourago': yesterday,
        '7dayago': week_ago,
        '1monthago': month_ago
    }

    sample = pd.DataFrame(sample, index = [0])
    prediction = model.predict(sample)[0]
    
    st.title(f"Predicted Bike Shares: {int(prediction)}")
    