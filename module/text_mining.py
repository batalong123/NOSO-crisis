import numpy as np
import pandas as pd
import streamlit as st 
import matplotlib.pyplot as plt
from module.clean_prepare import load_data
import seaborn as sns
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
import nltk 
from scipy.cluster.hierarchy import dendrogram, ward
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

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
def counts_media(data, cols):
	media = {}

	for u in data.YEAR.unique().tolist():
		df = data[data.YEAR == u]
		media[u] = {a: len(df[df.SOURCE == a].SOURCE.tolist()) for a in cols}
	return media

@st.cache(allow_output_mutation=True)
def text_preprocess(sents):
	wlem = nltk.WordNetLemmatizer()
	tokenizer = nltk.RegexpTokenizer('\w+')

	sents = sents.lower()
	sents = sents.strip()

	tokens = tokenizer.tokenize(sents)
	stopword = nltk.corpus.stopwords.words('english')

	return ' '.join([wlem.lemmatize(token) for token in tokens if token not in stopword and not(token.isdigit())])

@st.cache(allow_output_mutation=True)
def representation(data):
	agglo = AgglomerativeClustering(affinity='euclidean', distance_threshold=12,
		compute_distances=True, linkage='ward', n_clusters=None,
		compute_full_tree=True)
	nlp_pipe = Pipeline([('tfidf', TfidfVectorizer(sublinear_tf=True, min_df=2, max_df=1.0,
		ngram_range=(1, 1))), ('normalizer', Normalizer())])

	xmatrix = nlp_pipe.fit_transform(data)
	xmatrix = pd.DataFrame(xmatrix.toarray(), columns=nlp_pipe['tfidf'].get_feature_names_out())

	tsne = TSNE(n_components=2, random_state=0, n_jobs=-1, metric='cosine', 
            square_distances=True, early_exaggeration=20, learning_rate=200, init='random')
	Tmatrix = tsne.fit_transform(xmatrix.T)
	clusters = agglo.fit_predict(Tmatrix)

	return Tmatrix, clusters, nlp_pipe['tfidf'].get_feature_names_out()

def plot_word_network(data=None, c=None, feature_names=None):
	fig, ax = plt.subplots()

	colors = ['green', 'red', 'yellow',  'blue','black']
	for label, x, y, cx  in zip(feature_names, data[:, 0], data[:, 1], c):
		ax.text(x, y, label, bbox=dict(facecolor=colors[cx], alpha=0.8,
			boxstyle="round", fc='1.0'), fontweight='bold')
	ax.set_xlim(data[:, 0].min()-2, data[:, 0].max()+2)
	ax.set_ylim(data[:, 1].min()-2, data[:, 1].max()+2)
	ax.set_title('Word network')
	ax.set_xlabel('tsne_x')
	ax.set_ylabel('tsne_y')
	st.pyplot(fig)

def plot_dendrogram(data, labels):
	fig, ax = plt.subplots()
	linkage = ward(data)
	dendrogram(linkage, labels=labels, p=len(labels), 
		truncate_mode='level',ax=ax,
		 distance_sort=True, leaf_font_size=14)
	ax.set_xlabel('Word')
	ax.set_ylabel('Cluster distance')
	ax.set_title('hierarchical words')
	st.pyplot(fig)

def media_distribution(data):

	comment = """
		*Still today, Mimi Mefo and Undisclosed Source continue to give the news
		on anglophone crisis. All this two media are National.* *Let's see below 
		how many appearance each year, each media give the news on this crisis.* """     

	if st.checkbox("Media statistics"):

		df = data.SOURCE.value_counts()[:20]
		cols = df.index.tolist()[:12]
		media = pd.DataFrame(counts_media(data, cols))

		fig, ax = plt.subplots()
		df.plot(figsize=(15,8), kind='bar', 
			edgecolor='black', linewidth=2, ax=ax, legend=True)
		ax.set_title('20 most commons source')
		ax.set_ylabel('counts')
		ax.set_xlabel('Media')
		st.pyplot(fig)
		with st.expander('Comment'):
			st.markdown(comment)

		if st.button('How many source?'):
			n = data.SOURCE.nunique()
			st.write(f'The number of total source (media) is {n}.')
		if st.checkbox('Media appearances'):
			st.dataframe(pd.DataFrame(media))
			if st.button('plot'):
				fig, ax = plt.subplots()
				media.T.plot(legend=True,  ax=ax, linewidth=2)
				ax.set_title('Media appearance curves')
				ax.set_xlabel('Year')
				ax.set_ylabel('Appearances')
				st.pyplot(fig)

@st.cache(allow_output_mutation=True)
def word_distribution(data):

	list_words = np.concatenate([data[i].split() for i in range(len(data.tolist()))])

	fdist_16 = nltk.FreqDist(list_words)
	filter_word = {w:fdist_16[w] for w in sorted(list(set(fdist_16.keys()) - set(fdist_16.hapaxes())))}
	filter_word = pd.Series(filter_word)

	ngram_16 = list(nltk.ngrams(list_words, 3))
	freq_ngrams_16 = nltk.FreqDist(ngram_16)
	filter_ngrams_16 = {w[0]+'_'+w[1]+'_'+w[2]+\
	f'({freq_ngrams_16[w]})':freq_ngrams_16[w] for w in sorted(list(set(freq_ngrams_16.keys())
		- set(freq_ngrams_16.hapaxes())))}

	filter_ngrams_16 = pd.Series(filter_ngrams_16)

	filter_word = filter_word.sort_values(ascending=False).reset_index()
	filter_ngrams_16 = filter_ngrams_16.sort_values(ascending=False).reset_index()

	filter_word.columns = ['Words', 'Frequency']
	filter_ngrams_16.columns = ['Group words', 'Frequency']



	return filter_word.style.background_gradient('Oranges') , filter_ngrams_16

def main_subject():
	pb1 = """ Fako land crisis"""
	pb2 = """Frustrations of Lawyers and teachers"""
	pb3 = """End discrimination against anglophone"""

	fig, ax = plt.subplots(figsize=(15, 5))
	ax.axis([0, 10, 0, 10])
	ax.text(1, 7, pb1, style='italic', weight="bold",
        bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 30})
	ax.text(3, 2, pb2, style='italic', weight="bold",
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 20})
	ax.text(6, 6, pb3, style='italic', weight="bold",
        bbox={'facecolor': 'green', 'alpha': 0.5, 'pad': 20})

	ax.axis(False)
	ax.set_title('The 3 main topics of the beginning of the anglophone crisis.')
	st.pyplot(fig)



def reconstitution(data):
	text_vector = np.vectorize(text_preprocess)
	data['EVENT_DATE'] = pd.to_datetime(data['EVENT_DATE'],
		errors='ignore')
	news = data[data.YEAR == 2016][['EVENT_DATE', 'NOTES']]
	news = news.groupby(by = ['EVENT_DATE'])['NOTES'].agg('sum')

	norm_text = text_vector(news)
	matrix, clusters, feature_names = representation(norm_text)
	data_f = pd.DataFrame(matrix, columns=['tsne_0','tsne_1'], index=feature_names)
	data_f['clusters'] = clusters

	read = """
		**`Semantic analysis`** is more about understanding 
		the actual context and meaning behind words in text 
		and how they relate to other words and convey some
		information as a whole.

		**`Clustering`** is the task of partitioning 
		the dataset into groups, called clusters. 
		The goal is to split up the data in such
		a way that points within a single cluster
		are very similar and points in different
		clusters are different.
		"""

	read1 = """
		**Word frequency** is the frequency at which a word appears a given text or corpus.

		**Collocation** helps identify words that commonly coexist (bigrams, trigrams)."""

	comment1 = """
		*This hierarchical clustering help us to see that we have 3 clusters.*
		*The words in the same cluster have same semantic. Each cluster have
		different semantic*. 

		*The first cluster (red color) can tell us that a news speak 
		a violence between security force (gendarme, police, etc...)
		and demonstrator (Lawyer, protester, etc..)*.

		*The second cluster (yellow color) can also tell us that a news speak a 
		march against the discrimination of anglophone and asking an independence*

		*The last cluster (green color) shows that a news speak about 
		the villager who are angry at a recent court ruling 
		involving land claims*
		"""

	comment2 = """
		*Word network help us to see how each word can go
		together and have same semantic.* 
			"""

	read2 = """
		The three main subjects in the news of year 2016 
		are: 
	"""


	if st.checkbox('Event reconstitution'):
		with st.expander('Read more'):
			st.markdown("""
				**What does the news of the year 2016 say about the beginning of NOSO crisis?**
			""")
		if st.checkbox('see/hide text events'):
			st.dataframe(news)

		if st.checkbox('Semantic analysis & Clustering'):
			with st.expander('Read more'):
				st.markdown(read)
			if st.checkbox('hierarchy'):
				plot_dendrogram(matrix, feature_names)
				with st.expander('Comment'):
					st.markdown(comment1)

			if st.checkbox('Word network'):
				plot_word_network(data=matrix, 
					c=clusters, feature_names=feature_names)
				with st.expander('Comment'):
					st.markdown(comment2)

	if st.checkbox('Word frequency and collocation'):
		with st.expander('Read more'):
			st.markdown(read1)
		word, ngrams_word = word_distribution(norm_text)

		if st.checkbox('Word'):
			st.dataframe(word)

		if st.checkbox('3-grams'):
			st.dataframe(ngrams_word)

	with st.expander("Conclusion"):
		st.markdown(read2)
		main_subject()







def news():
	text="""
	In this section, we will analyze the ratings of all
	the media that are interested in the anglophone crisis.
	We use a text mining to extract the knowledge.  
	"""
	st.title('Anglophone crisis news')
	st.markdown(text)

	noso = load_data()
	media_distribution(noso)
	reconstitution(noso)
