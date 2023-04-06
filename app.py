from flask import Flask, request, render_template
from flask_restful import Api
import spacy
from spacy import displacy
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_distances
from tqdm.notebook import tqdm
import dill as pickle
import string
import re
from bs4 import BeautifulSoup

nlp = spacy.load('en_core_web_md')

data_raw = pd.read_csv('Job Posting report.csv')
data_raw = data_raw.drop(['id', 'vacancies', 'division_id', 'country_id', 'contract_type', 'hierarchy_level', 'reference_code', 'district_id', 
             'gender', 'salary', 'effective_from', 'deadline', 'is_inside_only', 'created_by', 'updated_by', 'created_at', 
             'updated_at', 'status_id', 'hris_job_id', 'hris_position_id', 'company_id'], axis = 1)

app = Flask(__name__)
api = Api(app)

# class PredictSentiment(Resource):
#     def post(self):
#         args = parser.parse_args()
#         review = args['review']
#         doc = nlp(review)
#         score = doc._.polarity
#         return {'score': score}

# class HelloWorld(Resource):
#     def get(self):
#         return {'hello': 'world'}

# api.add_resource(HelloWorld, '/')
# api.add_resource(PredictSentiment, '/predict')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def extract():
    if request.method=='POST':
        # check = False
        # if check==False:
        #     render_template('loader.html')
        txt = request.form['text']
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

        candidate_data = pd.read_csv('candidate database.csv', low_memory=False)
        candidate_data = candidate_data[['id', 'candidate_name','skill_name']]
        candidate_data['similarities'] = 0.0

        # data['skills'] = ''

        for i in range(data.shape[0]):
            data.at[i, 'title'] = data.at[i, 'title'].lower()
            data.at[i, 'title'] = re.sub("[\(\[].*?[\)\]]", "", data.at[i, 'title'])
            data.at[i, 'title'] = data.at[i, 'title'].translate(str.maketrans('', '', string.punctuation))
        # data.to_csv('clean_title.csv')
        # data = pd.read_csv('clean_title.csv')
        vector = sent2vect(txt)

        loaded_model = pickle.load(open('knn_model.sav', 'rb'))
        output = loaded_model.predict(vector.reshape(1, -1))

        out = data.loc[data['title'] == output[0]]

        for index, row in out.iterrows():
            required_jd = row['description']

        vec = nlp(required_jd)
        # for i in range(len(candidate_data)):
        for i in range(1000):
            skill = nlp(candidate_data['skill_name'][i])
            candidate_data['similarities'][i] = skill.similarity(vec)

        candidates_sorted = candidate_data.sort_values(by=['similarities'], ascending=False)
        soup = BeautifulSoup(required_jd, features="html.parser")
        # check = True
        # desc_soup = soup.get_text(separator=' ')
        # print(soup.get_text(separator=' '))
        # print(candidates_sorted.head())

    return render_template('result.html', desc_soup=soup.get_text(separator=' '), title=out.iloc[0,0], desc=out.iloc[0,2], candidates=candidates_sorted) #title=title_html, desc=desc_html,

if __name__ == "__main__":
    app.run(debug=True) # port = 5000, 