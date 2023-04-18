from flask import Flask, request, render_template
from flask_restful import Api
import spacy
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_distances
import dill as pickle
import string
import re
from bs4 import BeautifulSoup
import joblib

nlp = spacy.load('en_core_web_md')

data = pd.read_csv('Job Posting report.csv')
data = data[['title', 'experience','description']]
for i in range(data.shape[0]):
    data.at[i, 'title'] = data.at[i, 'title'].lower()
    data.at[i, 'title'] = re.sub("[\(\[].*?[\)\]]", "", data.at[i, 'title'])
    data.at[i, 'title'] = data.at[i, 'title'].translate(str.maketrans('', '', string.punctuation))

candidate = pd.read_csv('candidate database.csv', low_memory=False)
candidate_data = candidate[['id', 'candidate_name','skill_name']]
candidate_data = candidate_data.assign(similarities=0.0)

# candidate_data['similarities'] = 0.0

def sent2vect(text):
    vector = np.zeros(300,)
    valid_tokens = 0
    for token in nlp(text):
        if not token.is_stop and not token.is_punct and token.has_vector:
            vector += token.vector
            valid_tokens += 1
    vector = vector/valid_tokens if valid_tokens > 1 else vector
    return vector

# def cosine_distance(v1, v2):
#     return cosine_distances([v1], [v2])[0]

# metric = cosine_distance

app = Flask(__name__)
api = Api(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def extract():
    if request.method=='POST':
        txt = request.form['text']
        # def sent2vect(text):
        #     vector = np.zeros(300,)
        #     valid_tokens = 0
        #     for token in nlp(text):
        #         if not token.is_stop and not token.is_punct and token.has_vector:
        #             vector += token.vector
        #             valid_tokens += 1
        #     vector = vector/valid_tokens if valid_tokens > 1 else vector
        #     return vector

        # data = pd.read_csv('Job Posting report.csv')
        # data = data[['title', 'experience','description']]

        # candidate_data = pd.read_csv('candidate database.csv', low_memory=False)
        # candidate_data = candidate_data[['id', 'candidate_name','skill_name']]
        # candidate_data['similarities'] = 0.0

        # for i in range(data.shape[0]):
        #     data.at[i, 'title'] = data.at[i, 'title'].lower()
        #     data.at[i, 'title'] = re.sub("[\(\[].*?[\)\]]", "", data.at[i, 'title'])
        #     data.at[i, 'title'] = data.at[i, 'title'].translate(str.maketrans('', '', string.punctuation))
        # data.to_csv('clean_title.csv')
        # data = pd.read_csv('clean_title.csv')
        # data = data.loc[:, ~data.columns.str.match('Unnamed')]
        
        vector = sent2vect(txt)

        # loaded_model = pickle.load(open('knn_model.sav', 'rb'))
        loaded_model = pickle.load(open('knn_model.pkl', 'rb'))
        # loaded_model = joblib.load('knn_model.joblib')
        output = loaded_model.predict(vector.reshape(1, -1))

        out = data.loc[data['title'] == output[0]]

        for index, row in out.iterrows():
            required_jd = row['description']

        vec = nlp(required_jd)
        # for i in range(len(candidate_data)):
        for i in range(1000):
            skill = nlp(candidate_data['skill_name'][i])
            candidate_data['similarities'][i] = skill.similarity(vec)
            # candidate_data.loc['similarities', i] = skill.similarity(vec)

        # for index, row in tqdm(candidate_raw.iterrows(), total=candidate_raw.shape[0]):
        #     skill = nlp(candidate_raw['skill_name'][index])
        #     candidate_raw['similarities'][index] = skill.similarity(vec)

        candidates_sorted = candidate_data.sort_values(by=['similarities'], ascending=False)
        soup = BeautifulSoup(required_jd, features="html.parser")

    return render_template('result.html', desc_soup=soup.get_text(separator=' '), title=out.iloc[0,0], desc=out.iloc[0,2], candidates=candidates_sorted) #title=title_html, desc=desc_html,

@app.route('/cv/', defaults={'id': None})
@app.route('/cv/<int:id>')
def show_CV(id):
    cv=candidate.loc[candidate['id'] == str(id)]
    cv = cv.loc[:, ~cv.columns.str.match('Unnamed')]
    return render_template('cv.html', cv=cv, len=cv.shape[1])

if __name__ == "__main__":
    app.run(port = 5000, debug = True) # port = 5000, debug=True 