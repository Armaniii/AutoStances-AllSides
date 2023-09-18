import re
import pandas as pd
import emoji


def decontract(phrase):
  phrase = re.sub(r"can\'t", "cannot", phrase)
  phrase = re.sub(r"won\'t", "will not", phrase)
  phrase = re.sub(r"let\'s", "let us", phrase)
  phrase = re.sub(r"n\'t", " not", phrase)
  phrase = re.sub(r"\'m", " am", phrase)
  phrase = re.sub(r"\'ll", " will", phrase)
  phrase = re.sub(r"\'re", " are", phrase)
  phrase = re.sub(r"\'d", " would", phrase)
  phrase = re.sub(r"\'ve", " have", phrase)
  phrase = re.sub(r"\'s", "", phrase)
  #"kpkb'ing"
  phrase = re.sub(r"\'ing", "", phrase)

  phrase = re.sub(r"can’t", "cannot", phrase)
  phrase = re.sub(r"won’t", "will not", phrase)
  phrase = re.sub(r"let’s", "let us", phrase)
  phrase = re.sub(r"n’t", " not", phrase)
  phrase = re.sub(r"’m", " am", phrase)
  phrase = re.sub(r"’ll", " will", phrase)
  phrase = re.sub(r"’re", " are", phrase)
  phrase = re.sub(r"’d", " would", phrase)
  phrase = re.sub(r"’ve", " have", phrase)
  phrase = re.sub(r"’s", "", phrase)
  #"kpkb'ing"
  phrase = re.sub(r"’ing", "", phrase)
  return phrase


def convert_emoji(phrase):
  phrase = re.sub(r"😢", " sad ", phrase)
  phrase = re.sub(r"🤨", " not confident ", phrase)
  phrase = re.sub(r"🙄", " annoying ", phrase)
  phrase = re.sub(r"😂", " laugh ", phrase)
  phrase = re.sub(r"💯", " 100 ", phrase)
  phrase = re.sub(r"🤔", " wondering ", phrase)
  phrase = re.sub(r"🤷", " i do not know ", phrase)
  phrase = re.sub(r"👊", " fist bump ", phrase)
  phrase = re.sub(r"🏼", " calendar ", phrase)
  phrase = re.sub(r"👌", " perfect ", phrase)
  phrase = re.sub(r"®", " reserved ", phrase)
  phrase = re.sub(r"🤦", " facepalm ", phrase)
  phrase = re.sub(r"♀️", " ", phrase)
  phrase = re.sub(r"⁰", " degree ", phrase)
  phrase = re.sub(r"😬", " nervous ", phrase)
  phrase = re.sub(r"🌿", " leaf ", phrase)
  phrase = emoji.demojize(phrase)
  return phrase


#function to convert all short-forms/ short terms
def clean_short(phrase):
  phrase = re.sub(r"fyi", "for your information", phrase)
  phrase = re.sub(r"tbh", "to be honest", phrase)
  phrase = re.sub(r" esp ", " especially ", phrase)
  phrase = re.sub(r" info ", "information", phrase)
  phrase = re.sub(r"gonna", "going to", phrase)
  phrase = re.sub(r"stats", "statistics", phrase)
  phrase = re.sub(r"rm ", " room ", phrase)
  phrase = phrase.replace("i.e.", " ")
  phrase = re.sub(r"idk", "i do not know", phrase)
  phrase = re.sub(r"haha", "laugh", phrase)
  phrase = re.sub(r"yr", " year", phrase)
  phrase = re.sub(r" sg ", " singapore ", phrase)
  phrase = re.sub(r" mil ", " million ", phrase)
  phrase = re.sub(r" =", " same ", phrase)
  phrase = re.sub(r" msr. ", " mortage serving ratio ", phrase)
  phrase = re.sub(r" eip ", " ethnic integration policy ", phrase)
  phrase = re.sub(r"^imo ", " in my opinion ", phrase)
  phrase = re.sub(r" pp ", " private property ", phrase)
  phrase = re.sub(r" grad ", " graduate ", phrase)
  phrase = re.sub(r" ns ", " national service ", phrase)
  phrase = re.sub(r" bc ", " because ", phrase)
  phrase = re.sub(r" u ", " you ", phrase)
  phrase = re.sub(r" ur ", " your ", phrase)
  phrase = re.sub(r"^yo ", " year ", phrase)
  phrase = re.sub(r" vs ", " versus ", phrase)
  phrase = re.sub(r" irl ", " in reality ", phrase)
  phrase = re.sub(r" tfr ", " total fertility rate ", phrase)
  phrase = re.sub(r" fk ", " fuck ", phrase)
  phrase = re.sub(r" fked ", " fuck ", phrase)
  phrase = re.sub(r" fucked ", " fuck ", phrase)
  phrase = re.sub(r".  um.", " cynical. ", phrase)
  phrase = re.sub(r" pre-", " before ", phrase)
  phrase = re.sub(r" ed ", " education ", phrase)
  phrase = re.sub(r"\”", " ", phrase)
  phrase = re.sub(r"\“", " ", phrase)
  phrase = re.sub(r"nn", " ", phrase)
  phrase = re.sub(r"-", "", phrase)
  phrase = re.sub(r"\*", "", phrase)
  return phrase


def remove_https(item):
  #remove https links
  item_1 = re.sub(r"[(+*)]\S*https?:\S*[(+*)]", "", item)
  #remove https links with no brackets
  item_2 = re.sub('http://\S+|https://\S+', " ", item_1)
  #remove link markups []
  #note that this will also remove comment fields with ["Delete"]
  item_3 = re.sub(r"[\(\[].*?[\)\]]", " ", item_2)

  return item_3


def clean(string1):
  """
Main Function for cleaning. The rest are helper functions.
Regex is used for a majority of the cleaning.

    """
  string = string1
  if isinstance(string, str):
    string = re.sub('(?:"([^,]*)")', '', string)
    string = re.sub('\\n', '', string)
    # string = re.sub(r"(&gt;).*\n", "", string)
    # string = re.sub(r"^[0-9]", " ", string)
    string = re.sub(r"&#x200B;", " ", string)
    # string = string.replace("[", "")

    # # string = remove_https(string)
    string = string.replace("nn", "")
    # string = re.sub(r"(\B'\b)|(\b'\B)", " ", string)
    # string = re.sub(r"…", " ", string)
    # string = re.sub(r"'", "", string)
    # string = re.sub(r"\"", "", string)

    # string = re.sub(r"\n", "", string)
    # # string = string.replace("\\", " ")
    # string = string.replace("/", " ")
    # #remove apostrophe at the beginning and end of each word (e.g. 'like, 'this, or', this')

    string = decontract(string)
    string = convert_emoji(string)
    string = clean_short(string)

    return string
  else:
    return "None"


def clean_master(df_orig):
  df = df_orig.copy()
  df['comment'] = df['comment'].apply(clean)
  return df

  # Option to increase speed by using multiprocessing:
  # from pandarallel import pandarallel

  # pandarallel.initialize()
  # df['cleaned_text'] = df['text'].parallel_apply(clean)
