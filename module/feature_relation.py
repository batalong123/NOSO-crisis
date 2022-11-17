import numpy as np
import pandas as pd
import streamlit as st 
import matplotlib.pyplot as plt
from module.clean_prepare import load_data
import seaborn as sns

plt.style.use('fivethirtyeight')
plt.rcParams['figure.edgecolor']='black'
plt.rcParams['figure.frameon'] = True
plt.rcParams['figure.autolayout'] = True
plt.rcParams['figure.figsize'] = (15, 7)
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.labelsize'] = "large"
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.titlepad'] = 10
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.edgecolor'] ='black'

text = """
	Let's define some notion
	- The **`correlation`** between several random variables
		is a notion of connection which contradicts their 
		independence. Correlation is not causation.

	- The **`causality`** is the influence by which an event,
		a process, a state or an object (a cause) contributes
		to the production of another event, process, state or object
		(an effect). This cause is partly responsible for the effect,  
		and the effect partly depends on the cause.

	- Statistical **`Mode`** or **`dominant value`** is the most 
		represented value of any variable in a given population.

	*Wikipédia*
	"""

def correlation_df(data):

	if st.checkbox('Feature correlation'):
		fig, ax = plt.subplots(figsize=(15, 10))
		sns.heatmap(data.corr(), annot=True, center=0, robust=True)
		plt.title('Correlation between characteristics variables')
		st.pyplot(fig)
		text1 = """
			*The `LATITUDE` and the `LONGITUDE` are correlated
			which means that to locate a fight between the army 
			of Cameroon and the Ambazonian rebel groups or an
			act of violence against civilians it is enough
			just to use the LATITUDE or the LONGITUDE.*
			"""
		with st.expander('Comment'):
			st.markdown(text1, unsafe_allow_html=False)

		df = data[['LATITUDE', 'LONGITUDE']]
		df.columns = ['lat', 'lon']
		if st.checkbox('map'):
			st.map(df)

def feature_engineering():

	st.title('Relation between features')
	st.markdown(text, unsafe_allow_html=False)

	noso = load_data()
	correlation_df(noso)