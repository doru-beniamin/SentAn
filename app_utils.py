# NLP Pkgs
import spacy
nlp = spacy.load("en_core_web_sm")
from spacy import displacy
from textblob import TextBlob
import pandas as pd 


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


# create a function to display the sentiment polarity
def display_sentiment_polarity(sentiment):
    polarity = sentiment.polarity
    
    # Interpret sentiment polarity
    if polarity > 0:
        polarity_label = "Positive"
    elif polarity < 0:
        polarity_label = "Negative"
    else:
        polarity_label = "Neutral"

    # Create a formatted string displaying sentiment analysis result
    result_string1 = f"Sentiment: {polarity_label},"
    result_string2 = f"Polarity score: {polarity}"

    return result_string1, result_string2

# create a function to display the sentiment subjectivity
def display_sentiment_subjectivity(sentiment):
    subjectivity = sentiment.subjectivity
    
    result_string3 = f"Subjectivity:  {subjectivity}"

    return result_string3
