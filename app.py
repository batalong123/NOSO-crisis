import streamlit as st
from PIL import Image
from module.contact import contact_me
from module.covid19_pandemic import welcome
from module.geospatial_analysis import geospatial_covid19
from module.variant_mutation import covid19_variant 


#to get connection
st.set_page_config(
page_title="Covid-19 assessment",
page_icon= ":smiley:",
layout="centered",
initial_sidebar_state="expanded")

file = 'image/batalongCollege.png'
image = Image.open(file)
img= st.sidebar.image(image, use_column_width=True)


st.sidebar.header('*-* Covid-19 assessment app *-*')
st.sidebar.text(""" 
Covid19 assessment app is 
a Data Science board which
allows an user to understand
and discover a knownledge
from Covid-19 pandemic data.
Be free to enjoy this app!.
    """)

st.sidebar.title('Section')
page_name = st.sidebar.selectbox('Select page:', ("Welcome", "Covid-19 geospatial analysis",
 "Variant and Mutation",
     "Contact")) #"Epidemiology terms", "USA Covid19",


if page_name == "Welcome":
	welcome()

if page_name == "Covid-19 geospatial analysis":
	geospatial_covid19()

if page_name == "Variant and Mutation":
	covid19_variant()

#if page_name == "Epidemiology terms":
#	pass

#if page_name == "USA Covid19":
	#pass

if page_name == "Contact":
	contact_me()