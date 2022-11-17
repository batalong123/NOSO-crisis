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

@st.cache(allow_output_mutation=True)
def eventstable(df):
	events = {}
	for u in df.YEAR.unique().tolist():
		data = df[df.YEAR == u]
		events[u] = {a: len(data[data.EVENT_TYPE == a].EVENT_TYPE.tolist()) for a in data.EVENT_TYPE.unique().tolist()}

	events = pd.DataFrame(events).fillna(0).T
	return events

@st.cache(allow_output_mutation=True)
def aggregate(data):
	data['EVENT_DATE'] = pd.to_datetime(data['EVENT_DATE'], errors="ignore")
	fatalities_year = data.dropna().groupby(by = ['YEAR'])['FATALITIES'].agg('sum')
	fatalities_days = data.dropna().groupby(by = ['EVENT_DATE'])['FATALITIES'].agg('sum')
	return data, fatalities_days, fatalities_year

def event_evolution(data):
	text="""
		# Evolution of the different type of events
		See how each distribution of event evolved since 
		the beginning of conflict.
		"""

	text1 = """
		*Correlation or causality. While some curves decrease (Riots and Protests) others
		increase (violence against civilian, battles).*

		*When we observe the curve violence against civilians and that of battles, we 
		have the impression that the two curves have the same monotony.* *Between 2018 and 2020,
		we see that the year 2019 is a minimun of the evolutive curve of Battles event and 
		Strategic developments event.* *This minimun can be explain by two points `The failure
		of big dialogue` and `UN Security Council General Assembly`.*"""

	comment = """
		The correlation between the variables shows us that:
		- *The events `Battles` and `Violence against civilians` have the same monotomy; 
			fighting between Cameroon's security forces and Ambazonian rebel groups
			spawns violence against civilians.*

		- *Riots correlate to Explosion/Remote violence*

		- *Battles lead to Strategic developments*
		"""

	if st.checkbox('Event type evolution'):
		with st.expander('Read more'):
			st.markdown(text)

		events = eventstable(data)
		cols = events.columns.tolist() #+ ['none']
		name = st.multiselect('Choose event(s):', cols[::-1])

		if len(name) == 0: #'none':
			st.warning('Choose an event (2 or 3 events are best).')
		else:
			fig, ax = plt.subplots()
			events[name].plot(kind='bar', ax=ax, edgecolor='black', linewidth=2, subplots=True)
			ax.set_title(f'{name} evolution between 2016-2021')
			ax.set_ylabel('Counts')
			ax.set_xlabel('Year')
			st.pyplot(fig)

			with st.expander('Comment'):
				st.markdown(text1)

		if st.checkbox('Correlation'):
			df = events.corr()

			st.dataframe(df.style.background_gradient('OrRd'))
			with st.expander('Comment'):
				st.markdown(comment)

def victims_days_years(data):
	text = """
			# Fatalities of the anglophone crisis by days and by years
			"""
	comment = """
		*Between 2016 and 2018 the number of victims increased rapidly.* 
		*In 2019, we observe a sudden decrease and then in 2020 the number
		of victims increases.* 

		- *The difference in casualties between 2018 and 2019 is 448
			(a considerable decrease).*

		- *The gap in casualties between 2019 and 2020 is 302
			(a worrying increase).*
		"""

	data, days, years = aggregate(data)
	if st.checkbox('Fatalities evolution'):
		with st.expander('Read more'):
			st.markdown(text)

		name = st.selectbox('Choose a period:', ['choose option', 'days', 'years'])

		if name == 'choose option':
			pass
		elif name == 'days':
			fig, ax = plt.subplots()
			days.dropna().plot(x='EVENT_DATE', y='FATALITIES', rot=30, legend=True)
			ax.set_title('DAILY FATALITIES ')
			st.pyplot(fig)
		else:
			fig, ax = plt.subplots()
			years.dropna().plot(x='EVENT_DATE', y='FATALITIES', rot=30, legend=True)
			ax.set_title('YEARLY FATALITIES ')
			st.pyplot(fig)
			with st.expander('Comment'):
				st.markdown(comment)

		if st.checkbox('Victims in the 13 departments'):
			admin2_yearly = data.pivot_table(values="FATALITIES",
                                    index="YEAR", columns="ADMIN2", aggfunc='sum')
			st.dataframe(admin2_yearly.fillna(0).T.style.background_gradient('Reds'))




















def time_series():
	text = """
		 We study how the anglophone crisis evolved from 2016 to 15 Oct.2021
		"""
	noso = load_data()
	st.title('Time series: correlation or causality')
	st.markdown(text)
	event_evolution(noso)
	victims_days_years(noso)