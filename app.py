import streamlit as st
from PIL import Image
from module.contact import contact_me
from module.nwsw_crisis import welcome
from module.edaviz import vizualisation
from module.feature_relation import feature_engineering
from module.evolution import time_series
from module.text_mining import news
from module.prediction import textclassification



#to get connection
st.set_page_config(
page_title="Cameroon-Anglophone crisis",
page_icon= ":smiley:",
layout="centered",
initial_sidebar_state="expanded")

file = 'image/batalongCollege.png'
image = Image.open(file)
img= st.sidebar.image(image, use_column_width=True)


st.sidebar.header('*-* Anglophone crisis DM app *-*')
st.sidebar.text(""" 
Anglophone crisis DM app is 
a Data Mining Board app which
allows an user to understand
and discover a knownledge
from the NOSO crisis database
powered by ACLED.
Be free to enjoy this app!.
    """)

st.sidebar.title('Section')
page_name = st.sidebar.selectbox('Select page:', ("Welcome", "EDA&Viz","Feature Relation",
 "Time series", "News mining", "Predict event_type", "Contact"))


if page_name == "Welcome":
	welcome()

if page_name == "EDA&Viz":
	vizualisation()

if page_name == "Feature Relation":
	feature_engineering()

if page_name == "Time series":
	time_series()

if page_name == "News mining":
	news()

if page_name == "Predict event_type":
	textclassification()


if page_name == "Contact":
	contact_me()