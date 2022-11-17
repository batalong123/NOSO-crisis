import pandas as pd 
import numpy as np 
import streamlit as st
import time


@st.cache(allow_output_mutation=True)
def load_data():
    data = pd.read_csv('data/NOSO_2016_2022_Jul29.csv')
    data.drop(columns=['ASSOC_ACTOR_1', 'ASSOC_ACTOR_2', 'TIMESTAMP',
                     'EVENT_ID_NO_CNTY', 'EVENT_ID_CNTY', 'ISO'], inplace=True)

    data.ACTOR2.fillna(' ', inplace=True)
    data.dropna(inplace=True)
    noso = data[(data.YEAR >= 2016) & (data.YEAR < 2022)]
    time.sleep(2)
    return noso