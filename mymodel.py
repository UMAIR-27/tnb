import streamlit as st
import pandas as pd
import pickle

from PIL import Image

Image.open('tnb_logo.png').convert('RGB').save('tnb_logo.jpeg')
im = Image.open("tnb_logo.jpeg")
st.image(im, width = 300)

col1, col2 = st.columns (2)

df = pd.read_csv('electricity_usage.csv')


with col1:
    df_city = df.groupby('City')['MonthlyHours'].sum().reset_index()
    st.bar_chart(df_city, x="City", y="MonthlyHours")

with col2:
    df_city2 = df.groupby('City')['ElectricityBill'].sum().reset_index()
    st.bar_chart(df_city2, x="City", y="ElectricityBill")

fan=st.slider("Number of Fan", 5,20,value=10)

fridge = st.slider("Number of refrigerator", 18,23,value=20)

aircon= st.slider("NUmber of AirCon", 0,3, value=3)

tv=st.slider("NUmber of TV", 5,20, value = 3)

hour = st.slider("number of TV", 390,600, value=400)

model = pickle.load(open("LRmodel.pkl", "rb"))

new_data={'Fan':[fan], 'Refrigerator':[fridge],'AirConditioner':[aircon],'Television':[tv],'MonthlyHours':[hour]}

pd.Dataframe.from_dict(new_data)

pred = model.predict(pd.DataFrame.from_dict(new_data))

pred = round(pred[0],2)

st.subheader('The predicted Electricity Bill is ' + str (pred))
