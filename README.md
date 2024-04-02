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

- **Finding Distance b/w Pickup & Destination Locations**

