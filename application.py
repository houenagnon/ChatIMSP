# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
import random
from gtts import gTTS
import os

# Initial Setup
nltk.download('stopwords')
nltk.download('wordnet')

# Read Data
df = pd.read_excel('./chatdb1.xlsx')
df.dropna(inplace=True)

# French stop words
french_stopwords = set(stopwords.words('french'))

def nltk_tokenizer(text):
    tokenizer = TreebankWordTokenizer()
    return tokenizer.tokenize(text)

def preprocess(text):
    text = text.replace("\n", " ")
    text = text.replace("’", "e ")
    text = text.replace("'", "e ")

    text_tokens = nltk_tokenizer(text)
    text_tokens = [token.lower() for token in text_tokens]
    text_tokens = [token for token in text_tokens if token not in french_stopwords]
    text_tokens = [token for token in text_tokens if token not in list(string.punctuation)]

    lemmatizer = SnowballStemmer("french")
    text_tokens = [lemmatizer.stem(word) for word in text_tokens]

    processed_text = ' '.join(text_tokens)
    return processed_text

df['Questions1'] = df['Questions'].apply(preprocess)
df['Réponses1'] = df['Réponses'].apply(preprocess)

vectorizer = TfidfVectorizer()
vectorizer.fit(np.concatenate((df.Questions, df.Réponses)))
Question_vectors = vectorizer.transform(df.Questions)

DEFAULT_RESPONSES = ["Désolé, je ne comprends pas.", "Pouvez-vous reformuler ?",
                     "Je ne suis pas sûr de comprendre.", "Pardon, je ne suis pas programmé pour répondre à ça.",
                     "Je crains de ne pas avoir d'informations à ce sujet.", "Je ne peux pas répondre à cette question spécifique.",
                     "Ma capacité à répondre à cela est limitée.", "Je suis désolé, cela dépasse mes capacités actuelles.",
                     "Il se peut que je ne dispose pas d'informations sur ce sujet.", "Cette question est en dehors de mes compétences."]

GOODBYE_INPUTS = ("au revoir", "à bientôt", "soutien", "à la prochaine", "ciao", "à plus tard", "à demain", "bye",
                  "adieu", "à tout à l'heure", "bonne journée", "bonne soirée", "prends soin de toi",
                  "à la revoyure", "adios", "à la semaine prochaine", "à la rentrée", "à la revanche", "à la revue",
                  "à la reviviscence", "à l'avenir", "à l'an prochain", "ok", "je n'ai plus de question", "super",
                  "cool", "intéressant", "à plus")

END_RESPONSE = ["Je suis ravi de vous avoir aidé !", "Super alors. Si vous avez d'autres questions, n'hésitez pas !",
                "N'hésitez pas à revenir si vous avez d'autres questions. Au revoir !", "Cool ! Si vous avez besoin de plus d'informations, je suis là.",
                "Très bien ! Si quelque chose d'autre vous vient à l'esprit, faites-le moi savoir."]

WHO_ARE_YOU_INPUTS = ("qui es tu", "tu es qui", "c'est qui", "tu es quoi", "comment on t'appelle", "qui es tu", "qui t'a crée")

WHO_ARE_YOU_RESPONSES = ["Je suis un modèle de langage développé par les étudiants de Master2 DataScience de l'IMSP. Mon but est d'aider et de répondre à vos questions sur l'IMSP.",
                         "Je suis un assistant virtuel créé par les étudiants de Master2 DataScience de l'IMSP. Comment puis-je vous aider aujourd'hui ?",
                         "Je suis Chat IMSP, un programme informatique conçu pour comprendre et générer du texte en langage naturel.",
                         "Je suis une intelligence artificielle développée par les étudiants de Master2 DataScience de l'IMSP. En quoi puis-je vous assister ?",
                         "Je suis un modèle de langage avancé créé par les étudiants de Master2 DataScience de l'IMSP, ici pour vous aider avec vos questions et préoccupations."]

def whoAre(sentence):
    for phrase in WHO_ARE_YOU_INPUTS:
        if phrase.lower() in sentence.lower():
            return random.choice(WHO_ARE_YOU_RESPONSES)

def text_to_speech(text):
    tts = gTTS(text, lang='fr')
    tts.save('text_speech.mp3')
    with open('text_speech.mp3', 'rb') as audio_file:
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/mp3')
    os.remove('text_speech.mp3')

# Streamlit App
st.title("Assistant Chatbot IMSP")
st.write("Je suis prêt! Démarrons le chat...")

input_question = st.text_input("Vous :")
if input_question:
    if input_question.lower() not in GOODBYE_INPUTS:
        if whoAre(input_question):
            response = whoAre(input_question)
        else:
            input_question_processed = preprocess(input_question)
            input_question_vector = vectorizer.transform([input_question_processed])
            similarity = cosine_similarity(input_question_vector, Question_vectors)
            closest = np.argmax(similarity, axis=1)
            if closest == 0:
                response = random.choice(DEFAULT_RESPONSES)
            else:
                response = df['Réponses'].iloc[closest[0]]
        st.write(f"Imsp_Assistant: {response}")
        text_to_speech(response)
    else:
        response = random.choice(END_RESPONSE)
        st.write(f"Imsp_Assistant: {response}")
        text_to_speech(response)
