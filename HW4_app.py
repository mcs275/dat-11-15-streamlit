# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle

st.title("Predicting Fuel Economy for Different Vehicles by Car Characteristics")

st.subheader("Based on data from the EPA's fuel economy testing")

url = r"https://raw.githubusercontent.com/mcs275/dat-class-repo/main/Homework/Unit4/database.csv"

num_rows = st.sidebar.number_input('Select Number of Rows to Load', 
                                   min_value = 1000, 
                                   max_value = 38200, 
                                   step = 1000)

section = st.sidebar.radio('Choose Application Section', ['Data Explorer', 
                                                          'Model Explorer'])

@st.cache
def load_data(num_rows):
    df = pd.read_csv(url, nrows = num_rows)
    return df

##@st.cache
def create_grouping(x_axis, y_axis):
    grouping = df.groupby(x_axis)[y_axis].mean()
    return grouping

def load_model():
    with open('HW4_pipe2.pkl', 'rb') as pickled_mod:
        model = pickle.load(pickled_mod)
    return model

df = load_data(num_rows)

if section == 'Data Explorer':
    
    st.subheader("View average fuel economy in MPG for primary fuel type based on car's characteristics")
    
    x_axis = st.sidebar.selectbox("Choose column for X-axis", 
                                  df.select_dtypes(include = np.object).columns.tolist())
    
    y_axis = st.sidebar.selectbox("Choose column for y-axis", ['Combined MPG (FT1)'])
    
    chart_type = st.sidebar.selectbox("Choose Your Chart Type", 
                                      ['line', 'bar', 'boxplot'])

    
    if chart_type == 'line':
       grouping = create_grouping(x_axis, y_axis)
       fig = px.line(grouping)
       fig.update_layout(autosize=False, width=800, height=500)
       st.plotly_chart(fig)
        
    elif chart_type == 'bar':
       grouping = create_grouping(x_axis, y_axis)
       fig = px.bar(grouping, text_auto=True)
       fig.update_layout(autosize=False, width=800, height=500)
       px.title("Average Combined MPG")
       st.plotly_chart(fig)
        
    elif chart_type == 'boxplot':
        fig = px.box(df[[x_axis, y_axis]], x=x_axis, y=y_axis)
        fig.update_layout(autosize=False, width=800, height=500)
        px.title("Distribution of MPG values")
        st.plotly_chart(fig)
    
    st.write(df)
    
else:
    st.text("Choose Options to the Side to Predict Miles Per Gallon for Different Vehicles")
    model = load_model()
    
    
    year = st.sidebar.number_input("Enter Year of Car", 
                                min_value = 1984, max_value = 2022, step = 10, value = 2001)
    
    class_c = st.sidebar.selectbox("Choose class of car", 
                                     df['Class'].unique().tolist())
                                         
                              
    drive = st.sidebar.selectbox("Choose drive type for car", 
                                     df['Drive'].unique().tolist())

    cylinders = st.sidebar.number_input("Choose number of engine cylinders", 
                                  min_value=2, max_value =20, value=6, step=1)
   
    
    displacement = st.sidebar.number_input("Choose engine displacement",
                                       min_value=0.0, max_value =10.0, value=3.0, step=0.5)
    
    fuel = st.sidebar.selectbox("Select car's primary fuel type", 
                                       df['Fuel Type'].unique().tolist())
    
    alt_fuel = st.sidebar.selectbox("Select type of Alternative Fuel/Tech used by Car", 
                                       df['Alternative Fuel/Technology'].unique().tolist())
    
 
    sample = {
       'Year': year,
       'Class': class_c,
       'Drive': drive,
       'Engine Cylinders': cylinders,
       'Engine Displacement': displacement,
       'Fuel Type': fuel,
       'Alternative Fuel/Technology': alt_fuel
    }

    sample = pd.DataFrame(sample, index = [0])
    prediction = model.predict(sample)[0]
    
    st.title(f"Predicted Fuel Economy of Vehicle: {int(prediction)} Miles per Gallon")
    