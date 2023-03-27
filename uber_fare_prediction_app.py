# IMPORTING REQUIRED LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from math import *
import pickle # To load saved Machine Learning model,Scaler,One Hot Encoder
import streamlit as st
from streamlit_lottie import st_lottie # Used to import Lottie files
import streamlit_option_menu # Used for Navigation Bar
import requests
import json
from PIL import Image # Used to load images
from streamlit_extras.dataframe_explorer import dataframe_explorer

# SETTING PAGE CONFIGURATION
st.set_page_config(page_title='Uber Fare Prediction',layout='wide')
# SETTING STYLE
streamlit_style = """
			<style>
			@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@100&display=swap');

			html, body, [class*="css"]  {
			font-family: 'sans-serif';
			}
			</style>
			"""
st.markdown(streamlit_style, unsafe_allow_html=True)

# LOADING THE SAVED -- MODEL,ONE-HOT ENCODER,SCALER
loaded_model = pickle.load(open('uber_fare_RFprediction_model.sav','rb'))
loaded_scaler = pickle.load(open('uber_fare_scaler.sav','rb'))

# CREATING A FUNCTION THAT MAKES PREDICTION USING LOADED MODEL
def uber_fare_predictor(data):
    data_as_array = np.array(data)
    scaled_input_data = loaded_scaler.transform(data_as_array.reshape(1,-1))
    prediction = loaded_model.predict(scaled_input_data)
    return prediction[0]

def main():          
# USING LOCAL CSS
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)
    local_css('style.css')

# CREATING NAVIGATION BAR WITH OPTION_MENU    
selected = streamlit_option_menu.option_menu(menu_title=None,options=['About Data','Make Prediction'],icons=['activity','book'],menu_icon='list',default_index=0,orientation='horizontal',styles={
            "container": {"padding": "0!important", "background-color": "#white"},
            "icon": {"color": "yellow", "font-size": "25px"}, 
            "nav-link": {"font-size": "25px", "text-align": "middle", "margin":"0px", "--hover-color": "grey"},
            "nav-link-selected": {"background-color": 'blue'}
        })
    
# CREATING A PAGE TO GIVE INFORMATION ABOUT OUR PROJECT
# LOADING DATASET WITH PANDAS
df = pd.read_csv('uber.csv')
    
# CREATING PAGE FOR DATA DESCRIPTION
if selected=='About Data':
    st.write(' ')
    st.write(' ')
    st.image(Image.open('uber_taxi_image.jpeg'),width=1100)
    # TITLE
    st.title(':blue[Uber Fare Dataset]')
    st.write('''The project is about on world's largest taxi company Uber inc. In this project, we're looking to predict the fare for their future transactional cases. Uber delivers service to lakhs of customers daily. Now it becomes really important to manage their data properly to come up with new business ideas to get best results. Eventually, it becomes really important to estimate the fare prices accurately.''')
    st.write('''The dataset which we downloaded from kaggle contains 200000 rows and 6 columns.You can see some samples of that data below. ''')
    st.dataframe(dataframe_explorer(df.sample(2000,ignore_index=True).drop(['Unnamed: 0','key','pickup_datetime'],axis=1)),use_container_width=True)
    #st.write(df.sample(250,ignore_index=True))
    st.write('- - -')
    
    # DISPLAYING ATURES/COLUMNS IN DATASET
    st.subheader(':blue[Feature of Dataset]')
    st.write(' ')
    st.markdown(' - **Pick_up Longitude(in degrees)** : Longitude at which passengers are picked up by uber taxi driver.')
    st.markdown(' - **Pick_up Latitude(in degrees)** : Latitude at which passengers are picked up by uber taxi driver.')
    st.markdown(' - **Drop_off Longitude(in degrees)** : Longitude at which passengers are dropped  off by uber taxi driver.')
    st.markdown(' - **Drop_off Latitude(in degrees)** : Latitude at which passengers are dropped  off by uber taxi driver.')
    st.markdown(' - **Passenger Count** : Total number of passengers in the vehicle.')
    st.write('- - -')
    
    # DISPLAYING SOME PLOTS OF DATA
    
    st.subheader(':blue[Distribution Plots of Features]')
    
    # CREATING CONTAINER WITH 2 COLUMNS FOR 2 DIFFERENT PLOTS
    with st.container():
        plot_1,plot_2 = st.columns((1,1))
        with plot_1:
            st.write(' ')
            st.image(Image.open('dropoff_latitude_kdeplot.png'))
            st.markdown("<h1 style='text-align:center;font-size:20px '>Distribution Plot of Droppoff Latitude</h1>",unsafe_allow_html=True)
        with plot_2:
            st.write(' ')
            st.image(Image.open('dropoff_longitude_kdeplot.png')) 
            st.markdown("<h1 style='text-align:center;font-size:20px'>Distribution Plot of Droppoff Longitude</h1>",unsafe_allow_html=True)
    # CREATING CONTAINER WITH 2 COLUMNS FOR 2 DIFFERENT PLOTS
    with st.container():
        plot_1,plot_2 = st.columns((1,1))
        with plot_1:
            st.write(' ')
            st.image(Image.open('pickup_latitude_kdeplot.png'))
            st.markdown("<h1 style='text-align:center;font-size:20px'>Distribution Plot of Pickup Latitude</h1>",unsafe_allow_html=True)
        with plot_2:
            st.write(' ')
            st.image(Image.open('pickup_longitude_kdeplot.png')) 
            st.markdown("<h1 style='text-align:center;font-size:20px'>Distribution Plot of Pickup Longitude</h1>",unsafe_allow_html=True)     
    st.write('- - -')        
    st.markdown("<h5 color:Blue;font-size:10px'>Inference from above distribution plots:</h5>",unsafe_allow_html=True)
    st.markdown('- Wide range of dropoff_latitude column is in range between -10 and +10 degrees.')
    st.markdown('- Wide range of dropoff_longitude column is in range between -25 and +25 degrees.')
    st.markdown('- Wide range of pickup_latitude column is in range between -10 and +15 degrees.')
    st.markdown('- Wide range of pickup_longitude column is in range between -85 and -65 degrees.')        
            
    st.write('- - -')        
    st.markdown("<p class='font'>Procedure Followed</p>",unsafe_allow_html=True)
    st.write(' ')
    st.markdown('- We downloaded the dataset and loaded it in,by using Pandas.')
    st.markdown('- By using Python Libraries Numpy,Matplotlib and Seaborn,We did data exploration.')
    st.markdown('- We found distance between picked up location and drop off location by using corresponding longitude and latitude.')
    st.markdown('- We imported required libraries from Scikit Learn Library.')
    st.markdown("- Then,we made a train-test split on Features(X) and Target(y) of dataset.")      
    st.markdown('- We created instances of Regression algorithms and fitted them with training data.')
    st.markdown('- We compared metrics of Regression models using Error metrics like **Mean Absolute Error**,**Mean Squared Error**,**Root Mean Squared Error**.')
    st.markdown('- Finally,we selected best model with low error rates.')
    st.write('- - -')
     st.write('You can see Python code for Machine Learning [Here](https://github.com/TRGanesh/penguins_classification1/blob/main/penguins_DTClassification.ipynb)')
    st.write('You can see Python code for Streamlit web page [Here](https://github.com/TRGanesh/uber_fare_prediction/edit/main/uber_fare_prediction_app.py)')	

st.markdown(''' <style> .font{font-size:30px;
            font-weight:bold;
            font-family:'Copper Black';
            color:#FF9633;}</style>''',unsafe_allow_html=True)

# CREATING A PAGE FOR MODEL PREDICTION    
if selected=='Make Prediction':   
    # TITLE
    st.title(':blue[Uber Fare Prediction] :taxi:')
    st.write('You can give input features,that means pickup latitude,pickup longitude,dropoff latitude,dropoff longitude,passenger count as inputs and our Machine Learning model will predict the fare you have to pay.')
    # CONTAINER TO DISPLAY A FORM(TO TAKE INOUTS FROM USER) AND FARE
    with st.container():
        st.write('- - -')
        left_column,right_column = st.columns((2,1)) 
        with left_column:
            # GETTING DATA FROM USER	
            pickup_longitude = st.slider('**Pick_up Longitude**',-100.0,df['pickup_longitude'].max())
            pickup_latitude = st.slider('**Pick_up Latitude**',-80.0,50.0)
            dropoff_longitude = st.slider('**Drop_off Longitude**',-80.0,50.0)
            dropoff_latitude = st.slider('**Drop_off Latitude**',-80.00,50.0)
            passenger_count = st.selectbox("**Passenger Count**",[1,2,3,4,5,6])
        with right_column:
            st.write(' ');st.write(' ');st.write(' ')
            st.write(' ');st.write(' ');st.write(' ')
            st.write(' ');st.write(' ');st.write(' ')
            st.image(Image.open('uber_taxi_image2.png.jpeg'))
        
        # GETTING DISTANCE USING LONGITUDES AND LATITUDES
        def dist(longitude1, latitude1, longitude2, latitude2):
            long1,lati1,long2,lati2 = map(radians,[longitude1,latitude1,longitude2,latitude2])
            dist_long = long2 - long1
            dist_lati = lati2 - lati1
            a = np.sin(dist_lati/2)**2 + np.cos(lati1) * np.cos(lati2) * np.sin(dist_long/2)**2
            dis = 2 * asin(np.sqrt(a))*6371
            return dis
        distance = dist(pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude)    
        # DISPLAYING RESULT OUTSIDE OF CONTAINER
        st.markdown("<p class='font'>Model Prediction</p>",unsafe_allow_html=True)
        answer = ''
        # CREATING A BUTTON,ON CLICKING IT WE WILL BE ABLE TO SEE RESULT AS IMAGE
        if st.button('Result'):
            progress_bar = st.progress(0)
            for percentage_completed in range(100):
                time.sleep(0.0005)
                progress_bar.progress(percentage_completed+1)
            answer = uber_fare_predictor([passenger_count,distance])
            st.success(f'Predicted Uber Fare is ${np.round(answer,3)}',icon="✅")

if __name__ == "__main__":
    main()
    
    
