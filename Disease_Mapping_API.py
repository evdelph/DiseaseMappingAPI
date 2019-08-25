# Imports #

# Used to slow down time by 1 second to NOT overflow requests to NCBI #
import time as t

# Data Transformation #
import pandas as pd
import numpy as np

# Data Extraction Libraries #
import json
import lxml
import re
from bs4 import BeautifulSoup

# Used for advanced printing (Debugging) #
import pprint
pp = pprint.PrettyPrinter(depth=6)

# Used for NLP (entity extraction, stemming, and stopwords) #
import nltk
import spacy
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
nltk.download('wordnet')
from spacy.lang.en import English

# Used for fuzzy matching, uses Levenstein distance #
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
# This proxy code enables the requests to NCHBI databases #

# Make API Calls #
import urllib3

# For getting through Lilly's Proxy #
from urllib3 import ProxyManager

# Enables API calls to be made to NCBI Databases #
class NCBI_Authetication():
    
    def __init__(self):
        self.authenticate()
        
    def authenticate(self):
        self.base_url = "http://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.my_query = "PD-1%20ab%20agonist%5BTitle%2FAbstract%5D)%20AND%20(%222000%2F01%2F01%22%5BDate%20-%20Publication%5D%20%3A%20%223000%22%5BDate%20-%20Publication%5D"
        self.database = "pubmed"
        self.second_url = "esearch.fcgi?db={db}&term={query}&usehistory=y"
        self.final_url = self.base_url + self.second_url.format(db=self.database, query=self.my_query)
        self.http = ProxyManager("http://proxy.gtm.lilly.com:9000/")
        self.response = self.http.request('GET', self.final_url)
        self.http = ProxyManager("http://proxy.gtm.lilly.com:9000/")
        self.firstResponse = self.http.request('GET', self.final_url)
        
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.my_query = "id=29554659"
        self.database = "pubmed"
        self.second_url = "elink.fcgi?dbfrom=gene&db={db}&{query}"
        self.final_url = self.base_url + self.second_url.format(db=self.database, query=self.my_query)
        self.http = ProxyManager("http://proxy.gtm.lilly.com:9000/")
        self.secondResponse = self.http.request('GET', self.final_url)
        
    def get_response(self):
        return self.firstResponse,self.secondResponse

# Master Class for NCBI Database Object #

# Takes in a database #
class NCBI_Database_Call():
    
    def __init__(self,search_terms,threshold=1,score_threshold=70):
        self.search_terms = search_terms
        self.threshold = threshold
        self.score_threshold = score_threshold
        self.results = []
        self.mesh_call = None
        self.medgen_call = None
        self.df = pd.DataFrame()
        self.df_metadata = pd.DataFrame()
        
    def run(self):
        self.mesh_call = MeSH_Call(self.search_terms,self.threshold,self.score_threshold).query_terms(self.search_terms)
        self.medgen_call = Medgen_Call(self.search_terms,self.threshold,self.score_threshold).query_terms(self.search_terms)
        self.df, self.df_metadata = self.make_dataframe(self.mesh_call,self.medgen_call)
        return self.df,self.df_metadata
        
         
    # Abstract method for getting UID's in NCBI Databases #
    def get_uids(self): pass
    
    # Abstract method for getting metadata based on UID list #
    def get_terms(self): pass
    
    # Abstract method for writing a query search #
    def query_terms(self,terms): pass
    
    # Update results #
    def update_results(self):
        self.results.append(self.mesh_call[0])
        self.results.append(self.medgen_call[0])
        
    # Get results #
    def __repr__(self):
        return self.mesh_call
    
    def __repr__(self):
        return self.medgen_call
    
    def __repr__(self):
        return self.threshold
    
    # NLP Methods that will be used regardless of database selected #
    
    # Extracts only text, removing white spaces and punctuation #
    def preprocess(self,term):
        p = re.compile(r'\w+')
        return " ".join(p.findall(term)).strip().lower()
    
    # Lemmatizes words a search term #
    def lemmatize(self,term):
        processor = spacy.load("en")
        doc = processor(term)
        return " ".join([word.text if word.lemma_ == '-PRON-' else word.lemma_ for word in doc])
    
    # Stemming words in search term #
    def stem(self,term):
        term = term.split()
        sb_s = SnowballStemmer('english')
        return " ".join([sb_s.stem(word) for word in term])
    
    # Takes in the description and filters it by nounds and adjectives only #
    def extract_NN_JJ(self,description):
        description = nltk.word_tokenize(description)
        filtered_words = [nltk.pos_tag(word.split())[0][0] for word in description if nltk.pos_tag(word.split())[0][1] == 'NN' or nltk.pos_tag(word.split())[0] == 'JJ']
        return " ".join(filtered_words)
        
    # Calculates Jaccard Similarity for Description/Definition #
    def modified_jaccard_similarity(self,term, description):
        term = self.stem(self.preprocess(term)).split(" ")
        description = self.stem(self.lemmatize(self.preprocess(self.extract_NN_JJ(description)))).split(" ")
        intersection = list(set(term).intersection(description))
        return int((len(intersection)/len(term))*100)
    
    
    # Calculates Jaccard Similarity for Synonym and Disease #
    def jaccard_similarity(self,term, syn):
        term = term.split()
        syn = syn.split()
        intersection = list(set(term).intersection(syn))
        union = list(set(term).union(syn))
        return int((len(intersection)/len(union)*100))
    
    # Creates two dataframes, combined mesh/medgen and medgen meta_data #
    def make_dataframe(self,mesh,medgen):
        
        # Initial dataframe from mesh call #
        a = [y for x in mesh for y in x if len(y) > 0]
        d1 = pd.DataFrame(a)
        
        if not d1.empty:
            d1 = pd.DataFrame([d1.Disease_Input,d1.Ontology,d1.UID,d1.Ontology_ID,d1.Number_of_Results,d1.Synonym,d1.Synonym_Score,d1.Description_Score,d1.Final_Score,d1.Holder]).T
        
        # Intitial dataframe from medgen call #
        b = [y for x in medgen for y in x if len(y) > 0]
        d2 = pd.DataFrame(b)
        
        if not d2.empty:
            d_2 = pd.DataFrame([d2.Disease_Input,d2.Ontology,d2.UID,d2.Ontology_ID,d2.Number_of_Results,d2.Synonym,d2.Synonym_Score,d2.Description_Score,d2.Final_Score,d1.Holder]).T

            # Dataframe from medgen that has metadata #
            d_metadata = pd.DataFrame([d2.Disease_Input,d2.Ontology,d2.UID,d2.Ontology_ID,d2.Number_of_Results,d2.Semantic_Type,d2.SAB,d2.CODE,d2.SDUI,d2.SCUI,d2.Synonym,d2.Title,d2.Description,d2.Synonym_Score,d2.Description_Score,d2.Final_Score,d2.Holder]).T
        
        # Combine mesh and 1st medgen dataframe #
        d = d1.append(d_2)
        
        # Apply filtering to dataframes #
        
        if not d.empty:
            d = self.filter_dataframe(d)
            d_metadata = self.filter_dataframe(d_metadata)
        
        print('{0}'.format('Processing complete!'))
        
        return d,d_metadata
    
    # Filters/sorts dataframes #
    def filter_dataframe(self,df):
        
        # Drop duplicates #
        df = df.drop_duplicates()
        
        # Sort values #
        df = df.sort_values(['Disease_Input', 'Final_Score'], ascending=[True, False])
        
         # Get the top scores given the threshold #
        df = df.groupby('Disease_Input').head(self.threshold)
        
        # Remove UID field #
        df = df.drop('UID',1)
        
        # Remove Number_of_Results field #
        df = df.drop('Number_of_Results',1)
        
        # Drop Placeholder Score #
        df = df.drop('Holder',1)
        
        return df
 
 # Subclass MeSH_Call inherits from NCBI_Database_Call #
class MeSH_Call(NCBI_Database_Call):
    
    def __init__(self,search_terms,threshold,score_threshold):
        self.ontology = 'mesh'
        self.search_terms = search_terms
        self.threshold = threshold
        self.score_threshold = score_threshold
        self.count = 0
        self.mesh_results = []
        super()
    
    # Updates the count of a search result #
    def get_counts(self,number):
        self.count = number
    
    # Gets the count of a search result #
    def __repr__(self):
        return self.count
    
    # Get instance variable #
    def __str__(self):
        return str(self.ontology)
    
    # Update results list #
    def update_mesh_results(self,result):
        self.mesh_results.append(result)
        
    # Check if ID is associated with a disease by examining if it belongs to a C tree #
    def filter_by_disease(self,id,data):
        for tree in data["ds_idxlinks"]:
            if tree["treenum"][0] == 'C': return True
        return False

    # Retrieves the uids and count from mesh ontology #
    def get_uids(self,term):
        
        # Base Query and More Proxy Management #
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        term = self.preprocess(term).replace(" ","+")
        second_url = "esearch.fcgi?db={db}&term={query}&retmax=100&format=json"
        final_url = base_url + second_url.format(db=self.ontology, query=term)
        http = urllib3.PoolManager()
        http = ProxyManager("http://proxy.gtm.lilly.com:9000/")
        t.sleep(1)
                 
        # Response data #
        response = http.request('GET', final_url)
        json_data = json.loads(response.data)
        
        # Updates number of search results #
        self.get_counts(int(json_data['esearchresult']['count']))
        
        # Returns ID List #
        return json_data['esearchresult']['idlist']
      
    
    # Retrives a list of dictionaries where each dictionary is a score of a given disease #
    def get_terms(self,term,id,number_of_results):
        
        # Make API call to get json_data #
        term = self.lemmatize(self.preprocess(term))
        
        # It stores a given score result that will be added to scores, then to results #
        json_dict = dict()
        
        # Base Query and More Proxy Management # 
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        second_url = "esummary.fcgi?db=mesh&db=mesh&{query}&format=json"
        final_url = base_url + second_url.format(db=self.ontology, query="id="+id)
        http = urllib3.PoolManager()
        http = ProxyManager("http://proxy.gtm.lilly.com:9000/")
        t.sleep(1)
                 
        # Response data #
        response = http.request('GET', final_url)
        json_data = json.loads(response.data)
        uids = json_data['result']['uids']
 
        # Holds a list of dictionaries, will be converted to dataframe #
        results = []
        
        # Take the minimum of what the threshold is, versus the number of search hits #
        threshold = min(self.threshold,number_of_results)
         
        # Loop through each uid in the uids list #
        for uid in uids:
            
            # Keeps track of uids that score at or above the scoring requirement, used for pruning #
            counter = 0
            
            # This represents json data from the UID that is CURRENTLY being looped through #
            json_section = json_data['result'][uid]
            
            # Check if ID is a disease #
            check_id = self.filter_by_disease(id,json_section)
            
            # If the search term is a disease... #
            if check_id:
    
                # Pure extracted data from json file before processing #
                scope_note = json_section["ds_scopenote"]
                mesh_id = json_section["ds_meshui"]
                mesh_terms = json_section["ds_meshterms"]
                 
                # Intitialize score variables #
                score = None
                syn_score = None
                processed_term = self.stem(term)
                def_score = self.modified_jaccard_similarity(term,scope_note)
                
                # Keeps track of best scores for each uid #
                scores = []
        
                # If there's only one search result, take it (regardless of score), and return it #
                # Adding it to just the scores list is fine since it's the only output #
                if threshold == 1:
                    processed_mesh_term = self.stem(self.lemmatize(self.preprocess(mesh_terms[0])))
                    syn_score = fuzz.ratio(processed_mesh_term,processed_term) if len(processed_term.split()) == 1 and len(processed_mesh_term) == 1 else self.jaccard_similarity(processed_mesh_term,processed_term)
                    score = max(syn_score,def_score)
                    json_dict = {'Ontology':self.ontology,'UID':uid,'Ontology_ID':mesh_id,'Disease_Input':term,"Synonym":mesh_terms[0],"Description": scope_note,'Number_of_Results':number_of_results,'Synonym_Score':syn_score,'Description_Score':def_score,'Final_Score':syn_score + def_score,'Holder':score}
                    scores.append(json_dict)
                    return scores
             
                else:
                    
                    # Loop through each synonym in mesh_terms for scoring #
                    for mesh_term in mesh_terms:
                        
                        # Prepare synonymn for levenstein distance matching (through fuzzy library) #
                        processed_mesh_term = self.stem(self.lemmatize(self.preprocess(mesh_term)))
                        syn_score = fuzz.ratio(processed_mesh_term,processed_term) if len(processed_term.split()) == 1 and len(processed_mesh_term) == 1 else self.jaccard_similarity(processed_mesh_term,processed_term)
                   
                        # If term is only one word, just take the syn_score as its final score, otherwise take the max #
                        score = syn_score if len(term.split()) == 1 else max(syn_score,def_score)
                
                        # If the score is >= 60, add it to the scores list #
                        json_dict = {'Ontology':self.ontology,'UID':uid,'Ontology_ID':mesh_id,'Disease_Input':term,"Synonym":mesh_term,"Description": scope_note,'Number_of_Results':number_of_results,'Synonym_Score':syn_score,'Description_Score':def_score,'Final_Score':syn_score + def_score,'Holder':score}
                        scores.append(json_dict)
                  
                # This code takes scores, (as it has metadata for only ONE uid) and finds the best match #
                # Get the best score, if scores has results (it maybe empty) #
                if scores:
                    
                    # Gets the dictionary with the highest score and it's corresponding data #
                    best_score_data = max(scores, key=lambda x: x['Final_Score'])
                    best_score = best_score_data['Holder']
                    results.append(best_score_data)
                    
                    # If best score is greater than or equal to the threshold, increase counter (a step closer to threshold) #
                    if best_score >= self.score_threshold or threshold == 1:
                        counter += 1
                    
                    # If threshold is met, then return results #
                    if counter == threshold: 
                        return results
                    
        return results
    
      
    # Returns a list of dictionaries for future dataframe making #
    def query_terms(self,terms):
        
        # Check type is a list, check if it's a string #
        if type(terms) != list: terms = [terms]
            
        # Complete process for all terms in list #
        for disease in terms:
            
            # Get the uids #
            uids = self.get_uids(disease)
            uids = [uid for uid in uids if len(uid)]
            uids_string = " ".join(uids).replace(" ",",")
                    
            # For output message #
            message = 'result' if len(uids) == 1 else 'results'
            
            # Output message #
            print('{0} {1} {2} {3} {4}'.format('Processing MeSH term... ',disease,' has ',len(uids),' '+message))
            
            # Get number of search_results #
            number_of_results = len(uids)
           
            # Go through each UID #
            if number_of_results:
                result = self.get_terms(disease,uids_string,number_of_results)
   
                # Update results variable #
                if result: self.update_mesh_results(result)
    
        return self.mesh_results
        
class Medgen_Call(NCBI_Database_Call):
    
    def __init__(self,search_terms,threshold,score_threshold):
        self.ontology = 'medgen'
        self.search_terms = search_terms
        self.threshold = threshold
        self.score_threshold = score_threshold
        self.count = 0
        self.medgen_results = []
        super()
      
    # Get instance variable #
    def __str__(self):
        return str(self.ontology)
    
    # Updates the count of a search result #
    def get_counts(self,number):
        self.count = number
    
    # Gets the count of a search result #
    def __repr__(self):
        return self.count
    
    # Update results list #
    def update_medgen_results(self,result):
        self.medgen_results.append(result)
    
    # Medgen relies on webscraping just to get terms on the first page of results #
    def get_uids(self,term):
        
        base_url = "https://www.ncbi.nlm.nih.gov/medgen/?term="
        term = term.replace(" ","+")
        final_url = base_url + term
        http = urllib3.PoolManager()
        http = ProxyManager("http://proxy.gtm.lilly.com:9000/")
        response = http.request('GET', final_url)
        soup = BeautifulSoup(response.data,'lxml')
    
        pattern = "<dd>[0-9]*</dd>"
        p = re.compile(pattern)
        ids = p.findall(str(soup))
        ids = [id.replace("<dd>","").replace("</dd>","").strip() for id in ids]
        return ids

    
    # Get the metadata #
    def get_terms(self,term,id,id_string,number_of_results,is_match=False):
        
        # Make API call to get xml data #
        term = self.lemmatize(self.preprocess(term))
        
        # Proxy Code and Base Query #
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        second_url = "esummary.fcgi?db=medgen&db=medgen&{query}"
        final_url = base_url + second_url.format(db=self.ontology, query="id="+id_string)
        http = urllib3.PoolManager()
        http = ProxyManager("http://proxy.gtm.lilly.com:9000/")
        t.sleep(1)
        response = http.request('GET', final_url)
        soup = BeautifulSoup(response.data,'lxml')
        
        # Get the separate hits in lists #
        hits = soup.find_all('documentsummary')
        
        # Dictionary to store the results #
        results = []
        
        # Set threshold, take the min of the threshold requested and the total number of search results #
        threshold = min(self.threshold,number_of_results)
        
        # For every hit (each hit represents data from ONE UID) #
        for hit in hits:
            
            # Keeps track of meeting the threshold #
            counter = 0
            
            # Check if return is a disease #
            check = "Blank" if not len(hit.find("semanticid")) else hit.find("semanticid").text.strip()
    
            # List of acceptable semantic types #
            semantic_types = ['T191','T047','T048','T019','T190','T033','T049','T046','T184',"Blank"]
        
            # If term is a disease, execute the following: #
            if check in semantic_types:
                
                # Get Concept ID #
                concept_id = "Blank" if not len(hit.find('conceptid')) else hit.find('conceptid').text.strip()

                # Get Title #
                title = hit.find('title').text.strip()

                # Get name tags for looping #
                name_tags = hit.find_all('name')
        
                # Get definition/description #
                definition = hit.find('definition').text.strip()
                def_score = self.modified_jaccard_similarity(term,definition)

                # Get SAB, CODE, SCUI, SDUI, and Title #
                processed_term = self.stem(term)
                new_title = self.stem(self.lemmatize(self.preprocess(title)))
                
                # Keeps track of best scores for each uid #
                scores = []
            
                # Loop through synonyms #
                for data in name_tags:
                    
                    # Get the max syn_score between a synonym and the title #
                    new_text = self.stem(self.lemmatize(self.preprocess(data.text)))
                    syn_score = max(fuzz.ratio(new_text,processed_term),fuzz.ratio(processed_term,new_title))
                    syn_score = max(fuzz.ratio(new_text,processed_term),fuzz.ratio(processed_term,new_title)) if len(new_text.split()) == 1 and len(new_title.split()) == 1 and len(processed_term.split()) == 1 else self.jaccard_similarity(new_text,processed_term)
                    
                    # If score is 100 or the term is one word, take the syn_score #  
                    score = syn_score if len(term.split()) == 1 or syn_score == 100 else max(syn_score,def_score)
                
                    # Intialize dictionary to add to results #
                    value = dict()
                    code, sab, scui, sdui = None,None,None,None
                    index = hits.index(hit)
                    
                    # Add Basic Data MetaData to Dictionary #
                    value['Disease_Input'] = term
                    value['Ontology'] = self.ontology
                    value['Synonym'] = data.text
                    value['Description'] = definition
                    value['Semantic_Type'] = check
                    value['UID'] = id[index]
                    value['Ontology_ID'] = concept_id
                    value['Final_Score'] = syn_score + def_score
                    value['Synonym_Score'] = syn_score
                    value['Description_Score'] = def_score
                    value['Title'] = title
                    value['Number_of_Results'] = number_of_results
                    value['Holder'] = score
                
                    # Add extra metadata that may throw errors and add to dictionary #
                    try:
                        code = data['code']
                        value['CODE'] = code
                    except:
                        value['CODE'] = np.nan
                    try:
                        sab = data['sab']
                        value['SAB'] = sab
                    except:
                        value['SAB'] = np.nan
                    try:
                        scui = data['scui']
                        value['SCUI'] = scui
                    except:
                        value['SCUI'] = np.nan
                    try:
                        sdui = data['sdui']
                        value['SDUI'] = sdui
                    except:
                        value['SDUI'] = np.nan

                    scores.append(value)

                # This code takes scores, (as it has metadata for only ONE uid) and finds the best match #
                # Get the best score, if scores has results (it maybe empty) #
                if scores:
                    
                    # Gets the dictionary with the highest score and it's corresponding data #
                    best_score_data = max(scores, key=lambda x:x['Final_Score'])
                    best_score = best_score_data['Holder']
                    results.append(best_score_data)
                    
                    # If best score is greater than or equal to the threshold, increase counter (a step closer to threshold) #
                    if best_score >= self.score_threshold or threshold == 1: 
                        counter += 1
                    
                    # If threshold is met, then return results #
                    if counter == threshold:
                        return results
                
        return results
            
    # Returns a list of dictionaries for future dataframe making #
    def query_terms(self,terms):
        
        # Check type is a list, check if it's a string #
        if type(terms) != list: terms = [terms]
            
        # Complete process for all terms in list #
        for disease in terms:
            
            # Get the uids #
            uids = self.get_uids(disease)
            uids = [id for id in uids if len(id)]
            
            # stringify-ed ids for one big query search #
            uids_string = " ".join(uids).replace(" ",",")

            # For output message #
            message = 'result' if len(uids) == 1 else 'results'
            
            # Output message #
            print('{0} {1} {2} {3} {4}'.format('Processing Medgen term... ',disease,' has ',len(uids),' '+message))
           
            # Go through each UID #
            
            # Get the number of search results #
            number_of_results = len(uids)
            
            # Get the metadata #
            result = self.get_terms(disease,uids,uids_string,number_of_results)
            
            # If there's a result, add it to the list #
            if type(result) == list: 
                if result: self.update_medgen_results(result)
                    
        return self.medgen_results

# API function call provided incase user can't import module #
def api_input_setup(config_file):
    # Open the config file and convert it to json #
    with open(config_file) as f:
        data = json.load(f)
        f.close()
    
    # Save inputs to variables #
    file,threshold,scoring_threshold = data['file'],data['threshold'],data['scoring_threshold']

    # Validation message #
    print('{0},{1},{2}'.format('File Path: '+file,'\nUnique ID Threshold: '+str(threshold),'\nScore Threshold: '+str(scoring_threshold)))
    
    # Load file from config file #
    file = open(file)
    
    # Extract the list of disease inputs #
    search_list = [line.strip() for line in file.readlines()]
    
    # Return items needed to start the api call #
    return search_list,threshold,scoring_threshold

# Put your path HERE #
disease_inputs, threshold, scoring_threshold = api_input_setup("CONFIG FILE GOES HERE")
api_call = NCBI_Database_Call(disease_inputs,threshold,scoring_threshold)

# Run Code #
api_call.run()
