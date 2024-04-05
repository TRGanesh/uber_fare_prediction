# Uber Fare Prediction
### **Project Motive:**
By leveraging historical data from Uber trips, our aim is to create a Machine Learning Model that accurately predicts Uber fare amounts based on various Input Features such as Pickup and Dropoff Locations, and Number of Passengers.

---

### **Columns of DataSet:**
<pre>
<b>Unnamed: 0</b> : Record ID Number
<b>key</b> : Id generated based on DateTime 
<b>fare_amount</b> : Cost of an Uber Ride
<b>pickup_datetime</b> : Date & Time when the Uber Trip Begins
<b>pickup_longitude</b> : Geographic Longitude Co-Ordinate of the Location where an Uber Ride Begins
<b>pickup_latitude</b> : Geographic Latitude Co-Ordinate of the Location where an Uber Ride Begins
<b>dropoff_longitude</b> : Geographic Longitude Co-Ordinate of the Location where an Uber Trip Ends
<b>dropoff_latitude</b> : Geographic Latitude Co-Ordinate of the Location where an Uber Trip Ends
<b>passenger_count</b> : Number of Passengers in Trip
</pre>
---
**Sample DataSet**
<img width="902" alt="Screenshot 2024-04-02 at 11 47 01 PM" src="https://github.com/TRGanesh/uber_fare_prediction/assets/117368449/27f5fde8-d7c6-42b4-a346-86ed08d72322">


---
**Dependencies for Exploratory Data Analysis**
<pre>
<b>Pandas</b>
<b>Numpy</b>
<b>Matplotlib</b>
<b>Seaborn</b>
<b>Plotly</b>
<b>Statistics</b>
</pre>
---
**Data Pre-Processing & Analysis Steps:**
- Removed unnecessary columns such as Key, Unnamed 0 & PickUp Date Time
- Plotted a HeatMap which shows Correlation between the Numerical Columns

<img width="670" alt="Screenshot 2024-04-02 at 11 52 07 PM" src="https://github.com/TRGanesh/uber_fare_prediction/assets/117368449/9238dd58-dfaf-461f-959d-23c992c83239">

- **Calculated Distance b/w Pickup & Destination Locations by using Haversine Formula**
**Removal of UnWanted Records**
- Records where the Passenger Count is Greater than 6 were dropped
- Records where the Distance is 0 were dropped
---
**Feature Scaling**
```
- Feature Scaling is a Pre-Processing Technique used to Standardize the Range of Independent Variables or Features.
- It ensures that Features are on a Similar Scale, preventing variables with Larger scales from Dominating during Model Training.
- Common methods include
  - Normalization, which Scales Features to a Range b/w 0 & 1.
  - Standardization, which Rescales Features to have a Mean of 0 & a Standard Deviation of 1.
- It helps maintain the Stability and Effectiveness of Machine Learning Models across Different Features.
```
---
**Dependencies for Machine learning from Scikit-Learn**
1. **Train-Test Split :** Data is split into a Training Set (for Model Training) and a Test Set (for Model Evaluation), with a Common Split Ratio such as 70/30 or 80/20.
2. **StandardScaler :** Standardizes Features by making Mean to 0 & Standard Deviation to 1. Making them Comparable across Different Scales.
3. **Pickle :** Used to save the Machine Learning files, such as StandardScaler, OneHotEncoder.
---
**Models  used are**
<pre>
<b>Linear Regression</b>
  - Linear regression is a Fundamental Supervised Learning Algorithm used for Predictive Analysis.
  - It aims to find the Best-Fitting Linear Relationship between Independent Variables & a Target Variable.
  - It works by Minimizing the Residual Sum of Squares b/w Observed & Predicted Target Values.
  - Assumes a Linear Relationship between Variables and is Sensitive to Outliers.
<b>Decision Tree Regressor</b>
<b>Random Forest Regressor</b>
</pre>
