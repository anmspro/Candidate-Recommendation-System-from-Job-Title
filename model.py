import numpy as np
import pandas as pd
import spacy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_distances
from tqdm.notebook import tqdm
import dill as pickle
import string
import re
from bs4 import BeautifulSoup

nlp = spacy.load('en_core_web_md')

def sent2vect(text):
    vector = np.zeros(300,)
    valid_tokens = 0
    for token in nlp(text):
        if not token.is_stop and not token.is_punct and token.has_vector:
            vector += token.vector
            valid_tokens += 1
    vector = vector/valid_tokens if valid_tokens > 1 else vector
    return vector

data = pd.read_csv('Job Posting report.csv')
data = data.drop(['id', 'vacancies', 'division_id', 'country_id', 'contract_type', 'hierarchy_level', 'reference_code', 'district_id', 
             'gender', 'salary', 'effective_from', 'deadline', 'is_inside_only', 'created_by', 'updated_by', 'created_at', 
             'updated_at', 'status_id', 'hris_job_id', 'hris_position_id', 'company_id'], axis = 1)

data['skills'] = ''

for i in range(data.shape[0]):
    data.at[i, 'title'] = data.at[i, 'title'].lower()
    data.at[i, 'title'] = re.sub("[\(\[].*?[\)\]]", "", data.at[i, 'title'])
    data.at[i, 'title'] = data.at[i, 'title'].translate(str.maketrans('', '', string.punctuation))

new_job_desc = "software developer"

vector = sent2vect(new_job_desc)

loaded_model = pickle.load(open('knn_model.sav', 'rb'))
output = loaded_model.predict(vector.reshape(1, -1))

out = data.loc[data['title'] == output[0]]

for index, row in out.iterrows():
    print('id: ', index)
    print('Title: ', row['title'])
    print('Experience: ', row['experience'])
    print('Skills: ', row['skills'], '\n')
    print('Description: ', row['description'], '\n')