import re
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk import pos_tag

#!pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.0.0/en_core_web_sm-3.0.0.tar.gz
# import scattertext as st

nltk.download("averaged_perceptron_tagger")
nltk.download("omw-1.4")
nltk.download("wordnet")

try:
    nltk.data.find("punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("vader_lexicon")
except LookupError:
    nltk.download("vader_lexicon")


def stem_token(token):
    """
    Stems the given token using the PorterStemmer from the nltk library
    Input: a single token
    Output: the stem of the token
    """
    ps = PorterStemmer()
    stemmed_word = ps.stem(token)
    return stemmed_word


def penn2morphy(penntag):
    """Converts Penn Treebank tags to WordNet."""
    morphy_tag = {"NN": "n", "JJ": "a", "VB": "v", "RB": "r"}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return "n"


def lemmatize_token(token):
    """
    Lemmatize the token using nltk library
    Input: a single token
    Output: the lemmatization of the token
    """
    wordnet = WordNetLemmatizer()
    token_tagged = pos_tag([token])
    tag = token_tagged[0][1]
    morphy_tag = penn2morphy(tag)
    lemmatized_word = wordnet.lemmatize(token, pos=morphy_tag)
    return lemmatized_word


def remove_line_number(speech):
    """
    removes the line number at the beginning of speech

    Parameters
    ---------
    speech : str
        piece of text
    """

    pattern = "\n|^\d+.*?(\w)"
    speech = re.sub(pattern, "\n\g<1>", speech)
    pattern = "\t"
    speech = re.sub(pattern, "", speech)
    pattern = "\n\n"
    speech = re.sub(pattern, "\n", speech)
    pattern = "^\n *"
    speech = re.sub(pattern, "", speech)

    return speech


def filter_common_words(words):
    common_words = [
        "nations",
        "central",
        "new",
        "decision",
        "report",
        "and",
        "first",
        "like",
        "welcome",
        "pleased",
        "let",
        "good",
        "afternoon",
        "press",
        "conference",
        "meeting",
        "would",
        "outcome",
        "going",
        "know",
        "said",
        "along",
        "together",
        "also",
        "formally",
        "meetings",
        "evening",
        "annual",
        "one",
        "two",
        "second",
        "third",
        "last",
        "next",
        "point",
        "per",
        "answer",
        "ask",
        "say",
        "said",
        "mention",
        "talk",
        "tell",
        "told",
        "suggest",
        "think",
        "wonder",
        "mean",
        "understand",
        "know",
        "maybe",
        "perhaps",
        "remain",
        "generally",
        "thus",
        "member",
        "seem",
        "see",
        "look",
        "consider",
        "regard",
        "include",
        "hear",
        "going",
        "go",
        "goes",
        "come",
        "came",
        "give",
        "use",
        "using",
        "get",
        "can",
        "could",
        "should",
        "may",
        "might",
        "way",
        "yes",
        "no",
        "lot",
        "bit",
        "also",
        "case",
        "fact",
        "like",
        "want",
        "believe",
        "feel",
        "actual",
        "well",
        "kin",
        "moment",
        "time",
        "now",
        "current",
        "back",
        "ago",
        "make",
        "made",
        "today",
        "follow",
        "decided",
        "european",
        "take",
        "begin",
        "otherwise",
        "next",
        "met",
        "hold",
        "president",
        "excellency",
        "understand",
        "agreed",
        "addressed",
        "matter",
        "bodies",
        "decide",
        "tuesday",
        "recent",
        "year",
        "time",
        "relates",
        "word",
        "everyone",
        "precisely",
        "obviously",
        "united",
        "nations",
        "country",
        "countries",
        "state",
        "assembly",
        "today",
        "states",
        "something",
        "general",
        "people",
        "world",
        "peoples",
        "region",
        "international",
        "new",
        "especially",
        "proudly",
        "nothing",
        "must",
        "u",
        "thought",
        "consistently",
        "relevant",
        "many",
        "hopefully",
        "substantially",
        "ahead",
        "substantially",
        "however",
        "frankly",
        "mr",
        "great",
        "shall",
        "thank",
        "really",
        "warm",
        "congratulation",
    ]
    return [word for word in words if word not in common_words]


def remove_first_sentence(speech):
    """
    remove the first sentence which is always the welcoming
    """
    pattern = r"^.*?\."
    speech = re.sub(pattern, "", speech)

    return speech


def happiness_cleanup(happiness):
    """
    Creating a dictionary to map the wrong countries name in
    happiness dataframe with correct country name. This is because
    we need to merge happiness dataframe with speeches dataframe
    using country name. And we found cases where country names are
    not the same between these two data set.
    See function check_countryname_consistency as well.
    """

    country_mapping = {
        "Vietnam": "Viet Nam",
        "Moldova": "Republic of Moldova",
        "Laos": "Lao People's Democratic Republic",
        "Somaliland region": "Somalia",
        "Kosovo": None,
        "Taiwan Province of China": None,
        "United Kingdom": "United Kingdom of Great Britain and Northern Ireland",
        "United States": "United States of America",
        "South Korea": "Republic of Korea",
        "Ivory Coast": "Côte d’Ivoire",
        "Czech Republic": "Czechia",
        "Swaziland": "Eswatini",
        "Russia": "Russian Federation",
        "Hong Kong S.A.R. of China": "China-Hong Kong Special Administrative Region",
        "Palestinian Territories": "State of Palestine",
        "Tanzania": "United Republic of Tanzania",
        "Syria": "Syrian Arab Republic",
        "North Cyprus": None,
        "Bolivia": "Bolivia (Plurinational State of)",
        "Congo (Kinshasa)": "Democratic Republic of the Congo",
        "Venezuela": "Venezuela (Bolivarian Republic of)",
        "Iran": "Iran (Islamic Republic of)",
        "Congo (Brazzaville)": "Congo",
    }
    # Replace the country names in happiness dataframe with the correct country names in codes.
    happiness = (
        happiness.reset_index()
        .replace({"Country name": country_mapping})
        .set_index(["Country name", "year"])
    )
    return happiness


def country_code_cleanup(speech):
    """
    Replace YDYE (yemen) and POR (Portugal) with correct iso_alpho3 code.
    The remaing 'DDR', 'YUG', 'EU', 'CSK' are not considered countries by
    the UN or don't exist anymore, so we can consider removing them out of
    dataset because we don't have happiness data for these "countries".
    """
    speech["country"] = (
        speech["country"].str.replace("YDYE", "YEM").replace("POR", "PRT")
    )
    return speech


def preprocess_speech(speech):
    """
    This function does the preprocessing
    """
    # put all characters in lower case
    speech["Text"] = speech["Text"].str.lower()
    speech["Tokens"] = speech["Text"].apply(lambda x: nltk.word_tokenize(str(x)))
    # remove stop words and non-alphabetic from all the text
    stop_word = nltk.corpus.stopwords.words("english")
    speech["Tokens"] = speech["Tokens"].apply(
        lambda x: [word for word in x if (word not in stop_word) and word.isalpha()]
    )
    # lemmatizing
    speech["Tokens"] = speech["Tokens"].apply(
        lambda x: [lemmatize_token(token) for token in x]
    )
    # additional filter
    speech["Tokens"] = speech["Tokens"].apply(filter_common_words)
    speech["Joined_Tokens"] = speech["Tokens"].apply(lambda x: " ".join(x))
    speech = speech.sort_values(by="year").reset_index(drop=True)
    speech = country_code_cleanup(speech)
    # create a scattertext object for visualization
    # speech['parse'] = speech.Joined_Tokens.apply(st.whitespace_nlp_with_sentences)
    return speech


# read in country codes and happiness data
codes = pd.read_csv("Data/Raw/UNSD — Methodology.csv", delimiter=",")
happiness = pd.read_excel("Data/Raw/DataPanelWHR2021C2.xls", index_col=[0, 1])

# cleanup dataframes before merging
happiness = happiness_cleanup(happiness)

speech = pd.read_csv("Data/Raw/raw_speeches.csv", index_col=0)
speech = preprocess_speech(speech)

# merge speech with country code dataset
speech = speech.merge(codes, how="left", left_on="country", right_on="ISO-alpha3 Code")
# merge speech with happiness dataset
speech_happiness = pd.merge(
    speech,
    happiness,
    how="left",
    left_on=["year", "Country or Area"],
    right_on=["year", "Country name"],
)

speech_happiness.to_csv("Data/Processed/preprocessed_speech.csv")
