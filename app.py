from flask import Flask, request
from flask_restful import Resource, Api, reqparse
import spacy

nlp = spacy.load('en_core_web_md')

parser = reqparse.RequestParser()
parser.add_argument('review', required=True, help='Review cannot be blank!')

app = Flask(__name__)
api = Api(app)

# nlp = pickle.load(open('nlp.pkl', 'rb'))

class PredictSentiment(Resource):
    def post(self):
        args = parser.parse_args()
        review = args['review']
        doc = nlp(review)
        score = doc._.polarity
        return {'score': score}

class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}

api.add_resource(HelloWorld, '/')
api.add_resource(PredictSentiment, '/predict')

# @app.route('/')
# def home():
#     return render_template('index.html')


# @app.route('/',methods=['GET', 'POST'])
# def extract():
#     if request.method=='POST':
#         txt = request.form['ner_text']
#         doc=nlp(txt)
#         result = displacy.render(doc,style="ent")

#     return render_template('result.html',result=result)


if __name__ == "__main__":
    app.run()