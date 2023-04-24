import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv("cars_data.csv")
df = df.dropna()
df["MSRP"] = df["MSRP"].str.replace("$", "")
df["MSRP"] = df["MSRP"].str.replace(",", "")
df["MSRP"] = df["MSRP"].astype("int")
df_new = pd.get_dummies(df, columns= ['Make', 'Model', 'Type', 'Origin', 'DriveTrain'])
X = df_new.drop("MSRP", axis=1)
y = df_new["MSRP"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                   random_state=2)

# Train the model
model = LinearRegression()
model.fit(X, y)

# Train the model using Random Forest
random_model = RandomForestRegressor()
random_model.fit(X_train, y_train)
random_model.score(X_test, y_test)

# Train the model using Decision Tree
dec_model = DecisionTreeRegressor()
dec_model.fit(X_train, y_train)
dec_model.score(X_test, y_test)

# Define the Streamlit app
st.title("Car Price Predictor")
st.write("Enter the details of the car to get its estimated price")
# Write the model accuracy in the sidebar
st.sidebar.image("https://images-platform.99static.com/DpVhbKVn9kSF3AtQsV-SDdP8xIo=/174x174:1569x1569/500x500/top/smart/99designs-contests-attachments/81/81155/attachment_81155312", width=200)
st.sidebar.title("Live Model Accuracy Tracking")
st.sidebar.markdown("---")
st.sidebar.header("1. Linear Regression")
st.sidebar.write("Linear Regression Model Accuracy: ", round(model.score(X, y), 2))
st.sidebar.header("2. Random Forest Model")
st.sidebar.write("Random Forest Model Accuracy: ", round(random_model.score(X_test, y_test), 2))
st.sidebar.header("3. Decision Tree Model")
st.sidebar.write("Decision Tree Model Accuracy: ", round(dec_model.score(X_test, y_test), 2))
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://images.unsplash.com/photo-1478760329108-5c3ed9d495a0?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1974&q=80");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()

def add_sbg_from_url():
    st.markdown(
         f"""
         <style>
         .e1fqkh3o3 {{
             background-image: url("https://images.unsplash.com/photo-1584968153986-3f5fe523b044?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=987&q=80");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_sbg_from_url()


make = st.selectbox("Make", df["Make"].unique())
model_name = st.selectbox("Model", df[df["Make"]==make]["Model"].unique())
car_type = st.selectbox("Type", df[(df["Make"]==make) & (df["Model"]==model_name)]["Type"].unique())
origin = st.selectbox("Origin", df[(df["Make"]==make) & (df["Model"]==model_name) & (df["Type"]==car_type)]["Origin"].unique())
drive_train = st.selectbox("DriveTrain", df[(df["Make"]==make) & (df["Model"]==model_name) & (df["Type"]==car_type) & (df["Origin"]==origin)]["DriveTrain"].unique())
engine_size = st.slider("Engine Size", min_value=0.0, max_value=10.0, step=0.1)
cylinders = st.slider("Cylinders", min_value=2, max_value=12, step=1)
horsepower = st.slider("Horsepower", min_value=50, max_value=1000, step=10)
mpg_city = st.slider("MPG (City)", min_value=1, max_value=50, step=1)
mpg_highway = st.slider("MPG (Highway)", min_value=1, max_value=50, step=1)
weight = st.slider("Weight (lbs)", min_value=1000, max_value=8000, step=100)
wheelbase = st.slider("Wheelbase (inches)", min_value=50, max_value=200, step=1)
length = st.slider("Length (inches)", min_value=100, max_value=300, step=1)

# Make prediction
car = np.zeros(len(X.columns))
car[X.columns.get_loc("EngineSize")] = engine_size
car[X.columns.get_loc("Cylinders")] = cylinders
car[X.columns.get_loc("Horsepower")] = horsepower
car[X.columns.get_loc("MPG_City")] = mpg_city
car[X.columns.get_loc("MPG_Highway")] = mpg_highway
car[X.columns.get_loc("Weight")] = weight
car[X.columns.get_loc("Wheelbase")] = wheelbase
car[X.columns.get_loc("Length")] = length

make_col = "Make_" + make
model_col = "Model_" + model_name
type_col = "Type_" + car_type
origin_col = "Origin_" + origin
drive_train_col = "DriveTrain_" + drive_train

if make_col in X.columns:
    car[X.columns.get_loc(make_col)] = 1
if model_col in X.columns:
    car[X.columns.get_loc(model_col)] = 1
if type_col in X.columns:
    car[X.columns.get_loc(type_col)] = 1
if origin_col in X.columns:
    car[X.columns.get_loc(origin_col)] = 1
if drive_train_col in X.columns:
    car[X.columns.get_loc(drive_train_col)] = 1

#select the model to use for prediction
select_model = st.selectbox("Select the model to use for prediction", ["Linear Regression", "Random Forest", "Decision Tree"])
#if the selected model is Linear Regression
if select_model == "Linear Regression":
    prediction = model.predict([car])[0]
    #else if the selected model is Random Forest
elif select_model == "Random Forest":
    prediction = random_model.predict([car])[0]
    #else if the selected model is Decision Tree
elif select_model == "Decision Tree":
    prediction = dec_model.predict([car])[0]


#write the estimated price in a bold font and in green color
st.markdown("<h1 style='text-align: center; color: green;'>The estimated price of the car is: $"+str(round(prediction, 2))+"</h1>", unsafe_allow_html=True)