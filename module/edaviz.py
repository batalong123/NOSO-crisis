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


#@st.cache(allow_output_mutation=True)
def event_type(data):

	if st.checkbox('What type of events exist in the conflict?'):
		with st.expander('Read more'):
			text = """
				# Event type
				There exist six type of events in the anglophone crisis which are:

				1.**Battles**

				2.**Violence against civilians**

				3.**Riots**

				4.**Protests**

				5.**Strategic developments**

				6.**Explosions/Remote violence**

				Each event have its sub event for example **Battles** have as sub-event **Armed clash
				and gorverment regains territory**.
			"""
			st.markdown(text, unsafe_allow_html=False)

		fig, ax = plt.subplots()
		df = data.EVENT_TYPE.value_counts().sort_values(ascending=False)
		df.plot(kind='bar',
		ax=ax, edgecolor='black', linewidth=1.5)
		ax.set_title('EVENT TYPE')
		ax.set_ylabel('count')
		for i, u in enumerate(df):
			ax.text(i, u, str(u), fontsize=18, ha="center")
		st.pyplot(fig)
		with st.expander('Comment'):
			text = """
			*Between 2016 and 2021, we observe that the two major events are:
			***Battles*** and ***Violences against civilians***. These two events indicate
			the conflict in the noso region where the civilians are the first victims.*
			"""
			st.markdown(text, unsafe_allow_html=False)

def sub_event_type(data):
	battles = data[data.EVENT_TYPE=="Battles"]
	violence = data[data.EVENT_TYPE=="Violence against civilians"]
	strategic = data[data.EVENT_TYPE=="Strategic developments"]
	protester = data[data.EVENT_TYPE=="Protests"]
	riots = data[data.EVENT_TYPE=="Riots"]
	explosion = data[data.EVENT_TYPE=="Explosions/Remote violence"]

	if st.checkbox('The sub event of event'):
		fig, ax = plt.subplots()

		event_list = ['none',"Battles","Violence against civilians",
			"Strategic developments", "Protests", "Riots", "Explosions/Remote violence"]
		name = st.selectbox('Choose event:', event_list)

		if name == event_list[0]:
			st.warning('Choose an event to plot its sub events.')
		elif name == event_list[1]:

			battles["SUB_EVENT_TYPE"].value_counts().plot(kind="bar", ax=ax,
				edgecolor='black', linewidth=2)
			for i, u in enumerate(battles["SUB_EVENT_TYPE"].value_counts()):
			    ax.text(i, u, str(u), fontsize=18, ha="center")
			ax.set_title("Battles sub event")
			ax.set_ylabel('counts')
			st.pyplot(fig)

		elif name == event_list[5]:
			riots["SUB_EVENT_TYPE"].value_counts().plot(kind="bar", ax=ax,
				edgecolor='black', linewidth=2)
			for i, u in enumerate(riots["SUB_EVENT_TYPE"].value_counts()):
			    ax.text(i, u, str(u), fontsize=18, ha="center")
			ax.set_title("Riots sub event")
			ax.set_ylabel('counts')
			st.pyplot(fig)

		elif name == event_list[6]:	
			explosion["SUB_EVENT_TYPE"].value_counts().plot(kind="bar", ax=ax,
				edgecolor='black', linewidth=2)
			for i, u in enumerate(explosion["SUB_EVENT_TYPE"].value_counts()):
			    ax.text(i, u, str(u), fontsize=18, ha="center")
			ax.set_title("Explosions/Remote violence sub event")
			ax.set_ylabel('counts')
			st.pyplot(fig)

		elif name == event_list[4]:

			protester["SUB_EVENT_TYPE"].value_counts().plot(kind="bar", ax=ax,
				edgecolor='black', linewidth=2)
			for i, u in enumerate(protester["SUB_EVENT_TYPE"].value_counts()):
			    ax.text(i, u, str(u), fontsize=18, ha="center")
			ax.set_title("Protesters sub event")
			ax.set_ylabel('counts')
			st.pyplot(fig)

		elif name == event_list[2]:

			violence["SUB_EVENT_TYPE"].value_counts().plot(kind="bar", ax=ax,
				edgecolor='black',linewidth=2)
			for i, u in enumerate(violence["SUB_EVENT_TYPE"].value_counts()):
			    ax.text(i, u, str(u), fontsize=18, ha="center")
			ax.set_title("Violence against civilians sub event")
			ax.set_ylabel('counts')
			st.pyplot(fig)
		else:

			strategic["SUB_EVENT_TYPE"].value_counts().plot(kind="bar", ax=ax,
				edgecolor='black', linewidth=2)
			for i, u in enumerate(strategic["SUB_EVENT_TYPE"].value_counts()):
			    ax.text(i, u, str(u), fontsize=18, ha="center")
			ax.set_title("Strategic development  sub event")
			ax.set_ylabel('counts')
			st.pyplot(fig)
		with st.expander('Read more'):
			text = """
			The ACLED code: Sub event and event description can be taken from this
			pdf file. [codebook ACLED](https://reliefweb.int/sites/reliefweb.int/files/resources/ACLED_Codebook_2017FINAL%20%281%29.pdf).
			"""
			st.markdown(text, unsafe_allow_html=False)

def actors(data):
	if st.checkbox('Actors in the conflict'):

		actor1 = data.ACTOR1.value_counts().sort_values(ascending=False).reset_index()
		actor2 = data.ACTOR2.value_counts().sort_values(ascending=False).reset_index()

		actor1.columns = ['ACTOR 1', 'COUNTS']
		actor2.columns = ['ACTOR 2', 'COUNTS']

		name = st.selectbox('Choose actor:', ['Actor1', 'Actor2'])

		if name == 'Actor1':
			st.dataframe(actor1)
		else:
			st.dataframe(actor2)


		text = """ 
				*This table shows how many time each actor in ACTOR 1 or ACTOR 2 are more 
				distributed in the anglophone crisis.*
				"""
		with st.expander('Comment'):
			st.markdown(text, unsafe_allow_html=False)  

def interaction(data):

	text = """
		# Type of actors and theirs interactions
		The different type of actors are coded as follows:
		- inter code 1: **States Forces**

		- inter code 2: **Rebel Group**

		- inter code 3: **Political Militias**

		- inter code 4: **Identity Militias**

		- inter code 5: **Rioters**

		- inter code 6: **Protesters**

		- inter code 7: **Civilians**

		- inter code 8: **External/others Forces**

		- inter code 0: **Non-violence**

		Actor 1 and Actor 2 interact in the conflict. The interaction code is the combination
		of two inter codes associated with the two main actors.

		The unique actor type codes are recorded in the **INTER1** and **INTER2** columns
		and the dialed number is recorded in the **INTERACTION** column.

		For example, if a country's army is fighting a political militia, and the codes 
		INTER1 and INTER2 are respectively **1** and **3**, the coumpound interaction
		is recorded as **13**. As follows, we are plotting a distribution of INTERACTION.
		"""  
	if st.checkbox('Type of actors and theirs interactions'):
		with st.expander('Read more'):
			st.markdown(text, unsafe_allow_html=False)

		fig, ax = plt.subplots()
		sns.countplot(x="INTERACTION", data=data, color='blue', ax=ax)
		ax.set_title('DISTRIBUTION OF INTERACTION CODE')
		st.pyplot(fig)

		text1 = """
			- *The state of Cameroon is fighting against rebel groups (INTERACTION CODE: 12).*
			
			- *The civilians are subjected to repression by Cameroonian security forces
				(INTERACTION CODE: 17).*
			
			- *Rebel groups attack civilians (INTERACTION CODE: 27 Civil War.)*
			"""
		with st.expander('Comment'):
			st.markdown(text1, unsafe_allow_html=False)

def places(data):
	text = """
	# Places of conflict
	The region where exist the anglophone crisis are North west and South west of Cameroon.
	The places of conflict of the Anglophone crisis will be subdivided into 
	Department (Admin 2) and District(Admin 3). We present the distribution of admin2, admin3 and location.
	"""
	df1 = data.ADMIN2.value_counts().sort_values(ascending=False).reset_index()
	df2 = data.ADMIN3.value_counts().sort_values(ascending=False).reset_index()
	df3 = data.LOCATION.value_counts().sort_values(ascending=False).reset_index()

	df1.columns = ['ADMIN2','COUNTS']
	df2.columns = ['ADMIN3','COUNTS']
	df3.columns = ['LOCATION','COUNTS']

	if st.checkbox('Places of conflict'):
		with st.expander('Read more'):
			st.markdown(text, unsafe_allow_html=False)

		name = st.selectbox('Choose a place:', ['none', 'ADMIN2','ADMIN3','LOCATION'])

		if name == 'none':
			st.warning('Choose a place to see a distribution')

		elif name == 'ADMIN2':
			st.dataframe(df1)
		elif name == 'ADMIN3':
			st.dataframe(df2)
		else:
			st.dataframe(df3)








def vizualisation():

	st.title('EDA and Vizualisation')

	noso = load_data()

	event_type(noso)
	sub_event_type(noso)
	actors(noso)
	interaction(noso)
	places(noso)
