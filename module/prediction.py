
import numpy as np
import joblib
import streamlit as st
import nltk
import pandas as pd

@st.cache(allow_output_mutation=True)
def text_preprocess(sents):
	wlem = nltk.WordNetLemmatizer()
	tokenizer = nltk.RegexpTokenizer('\w+')

	sents = sents.lower()
	sents = sents.strip()

	tokens = tokenizer.tokenize(sents)
	stopword = nltk.corpus.stopwords.words('english')

	return ' '.join([wlem.lemmatize(token) for token in tokens if token not in stopword and not(token.isdigit())])


def textclassification():
	text_vector = np.vectorize(text_preprocess)

	st.title('Prediction: what event is it?')

	st.markdown(
				"""
			Event are encoded like this:

			1. **0 --> Battles.**
			2. **1 --> Explosions/Remote violence.**
			3. **2 --> Protests.**
			4. **3 --> Riots.**
			5. **4 --> Strategic developments.**
			6. **5 --> Violence against civilians.**
				""")
	file = 'models/Event_type_classification.pkl'
	classifier = joblib.load(file)# classifier

	notes = []
	nb_text = st.number_input('Give the number of text here:',1)	

	for i in range(int(nb_text)):
		notes.append(st.text_area('Write your text here:', key=i)) # note write by source media.
		
	note_vectorized = text_vector(np.array(notes))

	if st.checkbox('read your text'):
		st.write(np.array(notes))

	label = ['Battles', 'Explosions/Remote violence',
	 				'Protests', 'Riots', 'Strategic developments',
	 				'Violence against civilians']
	if st.button('predict'):
		#ext = os.path.splitext(raw_file)[1]
		if nb_text > 1:

			proba = classifier.predict_proba(note_vectorized)
			textp = ['probability of text'+str(i) for i in range(int(nb_text))]
			prob = pd.DataFrame(proba, columns=label, index=textp)
			st.success('Prediction is ok, see probability.')
			st.dataframe(prob)

		else:

			pred = classifier.predict(note_vectorized)
			proba = classifier.predict_proba(note_vectorized)
			res = f'Event is {label[pred[0]]} with the probability of {100*proba[0][pred][0]}%'

			prob = pd.DataFrame(100*proba[0], columns=['probability(%)'], index=label)
			st.success(res)
			st.dataframe(prob)
