# Core Pkgs
import streamlit as st
import streamlit.components.v1 as stc

# EDA Pkgs

import pandas as pd


# NLP Pkgs
import spacy
nlp = spacy.load("en_core_web_sm")
from spacy import displacy
from textblob import TextBlob


# Text Cleaning Pkgs
import neattext as nt 
import neattext.functions as nfx



# Fxns

## create a text analyzer function
def text_analyzer(my_text):
	docx = nlp(my_text)
	allData = [(token.text, token.shape_, token.pos_, token.tag,
		token.lemma_, token.is_alpha, token.is_stop) for token in docx]

	df = pd.DataFrame(allData, 
		columns=['Token', 'Shape', 'PoS', 'Tag', 'Lemma', 'IsAlpha', 'IsStopword'])

	return df


# create a function to get the entities:
def get_entities(my_text):
	docx = nlp(my_text)
	entities = [(entity.text, entity.label_) for entity in docx.ents]
	return entities



# create a function to display the entities
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 5px; padding: 10px; margin: 10px;">{}</div>"""
# @st.cache
def render_entities(rawtext):
	docx = nlp(rawtext)
	html = displacy.render(docx,style='ent')
	html=html.replace("\n\n","\n")
	result = HTML_WRAPPER.format(html)
	return result



# create a function to get the sentiment
def get_sentiment(my_text):
	blob = TextBlob(my_text)
	sentiment = blob.sentiment
	return sentiment


def main():
	st.title('NLP app with Streamlit')
	menu = ['Home', 'NLP', 'About']

	choice = st.sidebar.selectbox('Menu', menu)

	if choice == 'Home':
		st.subheader('Home: Analyze Text')
		raw_text = st.text_area("Enter text Here")
		num_most_common = st.sidebar.number_input("Most Common Tokens", 5, 15 )

		if st.button("Analyze"):

			with st.expander('Original Text'):
				st.write(raw_text)

			with st.expander('Text Analysis'):
				token_result_df = text_analyzer(raw_text)
				st.dataframe(token_result_df)

			with st.expander('Entities'):
				# entity_result = get_entities(raw_text)
				# st.write(entity_result)

				entity_result = render_entities(raw_text)
				stc.html(entity_result, scrolling=True)


			# Layout
			col1, col2 = st.columns(2)

			with col1:
				with st.expander("Sentiment"):
					sent_results = get_sentiment(raw_text)
					st.write(sent_results)

			with col2:
				with st.expander("Plot Wordcloud"):
					st.write(raw_text)




	elif choice == "NLP":
		st.subheader('NLP Task')

	else:
		st.subheader("About")







if __name__ == '__main__':
	main()

