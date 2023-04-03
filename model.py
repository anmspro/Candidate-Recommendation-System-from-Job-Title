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



