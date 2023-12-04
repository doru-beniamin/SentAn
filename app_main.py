# Core Pkgs
import streamlit as st
import streamlit.components.v1 as stc

# EDA Pkgs

import pandas as pd


# Text Cleaning Pkgs
import neattext as nt 
import neattext.functions as nfx

# External Utils
from app_utils import *




def main():
	st.title('NLP app with Streamlit')
	menu = ['Home', 'About']

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
				entity_result = render_entities(raw_text)
				stc.html(entity_result, scrolling=True)


			# Layout
			col1, col2 = st.columns(2)

			with col1:
				with st.expander("Sentiment"):
					sent_results = get_sentiment(raw_text)
					st.write(sent_results)

			with col2:
				with st.expander("Sentiment Polarity"):
					sent_polarity = display_sentiment_polarity(sent_results)
					st.write(sent_polarity)

				with st.expander("Sentiment Subjectivity"):
					sent_subjectivity = display_sentiment_subjectivity(sent_results)
					st.write(sent_subjectivity)


	else:
		st.subheader("About")



if __name__ == '__main__':
	main()

