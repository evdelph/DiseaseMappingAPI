# Disease Mapping API Utilizing Natural Language Processing and NCBI Databases
This API attempts to utilize the data in NCBI databases to accurately map disease inputs to their corresponding ID's utilzing Natural Language Processing. This API currently is composed of two NCBI ontologies; MeSH (Medical Subject Headings) and MedGen. Each ontologys' result is compiled into a pandas dataframe and outputted to the user. Below will outline the API's instructions as well as a detailed explanation of the API's composition.

## Getting Started
These instructions will tell you what dependencies (libraries) you need to use the API as well as how to use it.

### Prerequisites
The API requires using spaCy and nltk libaries. If you are working on your local computer, make sure you have the permissions to download these libraries. If you are working on platforms like Juptyer, it's recommended to create a spacy virtual environment to run these libraries.

### Set-up
This will be done in **config_file.json**. Configure the three variables by inputting the file path of **YOUR** file that contains disease inputs, input a threshold *(number of unique ontology ids)* you want for **EACH** disease input, and a minimum score you will accept.
```
{
    "file" :"YOUR FILE PATH HERE",
    "threshold" : YOUR THRESHOLD NUMBER HERE,
    "scoring_threshold" : YOUR SCORING THRESHOLD HERE
}
```
### Making an API Call
In API_Call.py OR Disease_Mapping_API.py, the only code you reconfigure is inputting the path to **config_file.json**. API_Call.py can be used if you are able to import other python files, whereas Disease_Mapping_API.py also contains the API_CALL.py code all in one file, incase you are unable to import python files.
```
disease_inputs, threshold, scoring_threshold = api_input_setup("YOUR CONFIG_FILE.json PATH HERE")
```
### Sample Call
If you are using API_Call.py to run the API, run it, otherwise run Disease_Mapping_API.py. Here is an example of what the disease inputs could be.
```
['Osteoarthritis of the knee (hospital diagnosed)',
 "Autoimmune thyroid diseases (Graves disease or Hashimoto's thyroiditis)",
 'Macrophage colony stimulating factor levels',
 'Incident coronary heart disease',
 'Cholesterol',
 'Colorectal cancer']
```
This is what the output looks like during the API calls.
```
Processing MeSH term...  Osteoarthritis of the knee (hospital diagnosed)  has  0  results
Processing MeSH term...  Autoimmune thyroid diseases (Graves disease or Hashimoto's thyroiditis)  has  0  results
Processing MeSH term...  Macrophage colony stimulating factor levels  has  0  results
Processing MeSH term...  Incident coronary heart disease  has  0  results
Processing MeSH term...  Cholesterol  has  100  results
Processing MeSH term...  Colorectal cancer  has  15  results
Processing Medgen term...  Osteoarthritis of the knee (hospital diagnosed)  has  0  results
Processing Medgen term...  Autoimmune thyroid diseases (Graves disease or Hashimoto's thyroiditis)  has  1  result
Processing Medgen term...  Macrophage colony stimulating factor levels  has  2  results
Processing Medgen term...  Incident coronary heart disease  has  0  results
Processing Medgen term...  Cholesterol  has  20  results
Processing Medgen term...  Colorectal cancer  has  20  results
Processing complete!
```
The API call outputs two dataframes, one combining the like results from MeSH and MedGen, and the other is composed of MedGen's **metadata**. In this example, ***scoring_threshold is 70 and threshold is 5***. This means for disease input, at **most** there will be 5 unique id's. Scoring_thershold is used to stop processing UIDs if we have reached threshold entries with a score at least 70. Below shows the first dataframe combining **both MeSH and MedGen**. 

<img src="https://user-images.githubusercontent.com/51026529/62747637-5e496200-ba23-11e9-8d53-d104065004a2.PNG" width="90%"></img> 

Below shows the second dataframe that has the **MedGen's metadata**. More information of the meta data mark-up (for SAB, SCUI, SDUI, and CODE) can be found [here](https://www.ncbi.nlm.nih.gov/books/NBK9685/table/ch03.T.concept_names_and_sources_file_mr/). Essentially, for each synonym, there COULD be another ontology id that ties to it. These extra ids could be incorporated for later mapping purposes.

<img src="https://user-images.githubusercontent.com/51026529/62747776-f34c5b00-ba23-11e9-86b9-626e7c5f54d0.PNG" width="90%"></img> 

## Explanation of the API's Code
Below outlines the API's code and how it works. This will help increase having a better understanding of the API in case more ontologies need to be added.

### Class Diagram
This is the class diagram for the API. It captures its methods and attributes. A free website called **diagrams.visual-paradigm** was used to make the class diagram. The link to this site is [here](https://diagrams.visual-paradigm.com/#proj=0&type=ClassDiagram).
<img src="https://user-images.githubusercontent.com/51026529/62702285-530b1d80-b9b4-11e9-8931-7af33d948a54.PNG" width="50%"></img>

### Disease Classification
MeSH and MedGen have two different classification schemes below outlines the differences between the two and how this was accounted for in the API's code. 

#### MeSH Disease Classification
For MeSH, anything that is a part of the *C Tree* is considered a disease. A given disease input can have many classification parent nodes. For our purposes, there must be *at least one treenum that starts with a **"C"*** for it to be accepted as a disease. An example of this is shown here, the *treenum* is the tree the disease input is associated with.
```
"ds_idxlinks": [
                {
                    "parent": 68009135,
                    "treenum": "C05.651.542",
                    "children": [
                    ]
                },
                {
                    "parent": 68009135,
                    "treenum": "C10.668.491.525",
                    "children": [
                    ]
                },
                {
                    "parent": 68059352,
                    "treenum": "C23.888.592.612.547.249",
                    "children": [
                    ]
                },
                {
                    "parent": 68059352,
                    "treenum": "F02.830.816.353.500",
                    "children": [
                    ]
                },
                {
                    "parent": 68059352,
                    "treenum": "G11.561.790.353.500",
                    "children": [
                    ]
                }
```
The filtering was implemented through the MeSH_Call's method **filter_by_disease**. Below shows its implementation.
```
def filter_by_disease(self,id,data):
        for tree in data["ds_idxlinks"]:
            if tree["treenum"][0] == 'C': return True
        return False
```
For more information regarding MeSH's classifcation click [here](https://www.nlm.nih.gov/mesh/intro_trees.html).
#### MedGen Disease Classification
MedGen has 13 different semantic categories. 10 out of the 13 were included as acceptable categories for disease classification. These semantic categories were analyzed by the Research Informatics team and included based on their discretion. The exception however was found with the semantic id TN00, which appeared to be a null value in disease input results' mark-up. Below shows an example of how the semantic id was stored in the xml mark-up. It is stored between the **SemanticId** tags.
```
<eSummaryResult>
<DocumentSummarySet status="OK">
<DbBuild>Build190802-0747.1</DbBuild>
<DocumentSummary uid="14319">
<ConceptId>C0027612</ConceptId>
<Title>
Congenital, Hereditary, and Neonatal Diseases and Abnormalities
</Title>
<Definition>
Diseases existing at birth and often before birth, or that develop during the first month of life (INFANT, NEWBORN, DISEASES), regardless of causation. Of these diseases, those characterized by structural deformities are termed CONGENITAL ABNORMALITIES.
</Definition>
<SemanticId>T047</SemanticId>
<SemanticType>Disease or Syndrome</SemanticType>
```
The filtering was implemented through a quick check in Medgen_Call's method **get_terms**. Below shows how the filtering was included.
```
# Check if return is a disease,TN00 is replaced with 'Blank' #
check = "Blank" if not len(hit.find("semanticid")) else hit.find("semanticid").text.strip()
    
# List of acceptable semantic types #
semantic_types = ['T191','T047','T048','T019','T190','T033','T049','T046','T184',"Blank"]
        
# If term is a disease, execute the following: #
if check in semantic_types:
    # execute rest of code #
```
For more information regarding Medgen's classifcation click [here](https://www.ncbi.nlm.nih.gov/medgen/docs/properties/).

### Getting UID's
MeSH and Medgen have different approaches getting the uids. Below outlines the two different approaches and why they were used.

#### Getting UID's Through MeSH
MeSH uses NCBI e-utilities (found [here](https://www.ncbi.nlm.nih.gov/books/NBK25501/)). More specifically, this call uses the **e-search** API call. This takes a disease input and returns a list of uids. Note that the timer module makes the call sleep for 1 second before executing, **this is due to NCBI's limit of 3 API calls per second**. The timer prevents an overflow of requests. The code to execute this is shown below:
```
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
```
#### Getting UID's Through MedGen
Due to the large number of search results per disease input, a webscraping approach was implemented rather than using e-utilities search API (since the UID's weren't always in order of the results as listed on the web page). Below shows the code used to webscrape NCBI:
```
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
```

### Basic Metadata Query Call for BOTH MeSH and Medgen
MeSH and Medgen have the same approach for making its API call to retrieve metadata. Both use NCBI's **e-summary** API call. All of the uids are converted into one long string, and sent through the API to get a results for **each** uid. The only difference is that MeSH is in a json format where Medgen is in xml format. Below is an example of this execution for MeSH.
```
# Response data #
response = http.request('GET', final_url)
json_data = json.loads(response.data)
uids = json_data['result']['uids']
         
# Loop through each uid in the uids list #
for uid in uids:
   # This represents json data from the UID that is CURRENTLY being looped through #
   json_section = json_data['result'][uid]
   # Go through rest of algorthim #
```
Here is how the same process is done using xml in Medgen.
```
# Get the separate hits in lists #
hits = soup.find_all('documentsummary')
# For every hit (each hit represents data from ONE UID) #
        for hit in hits:
        # Go through rest of algorthim #
```

### Scoring Algorithm
The **threshold and scoring_threshold** variables in **config_file** are the determiners for setting the precision of the results. The higher the threshold and lower the scoring threshold mean the fuzzier your results will be. Adjust those parameters as necessary. Below walks through the logic implemented for the scoring algorthim.

There are three different algorithms used to calculate the similarity scores of the synonyms and description. Below shows the break down:

* Synonyms One Word Long: **Levenstein Distance** (through the fuzzy libraries)
* Synonyms Longer Than One Word: **Jaccard Index** (manually implemented)
* Description: **Modified Jaccard Index** (manually implemented)

This is code for the jaccard index. It looks at the query and the synonym as sets. It calculates the commonalities of those two sets over the total number of unique terms between the two sets.
```
# Calculates Jaccard Similarity for Synonym and Disease #
def jaccard_similarity(self,term, syn):
    term = term.split()
    syn = syn.split()
    intersection = list(set(term).intersection(syn))
    union = list(set(term).union(syn))
    return int((len(intersection)/len(union)*100))
```

This is the code for the modified jaccard index. It looks at the similarity between the disease input and its description. It looks for the commonalities between the two over the number of terms in the disease input. Here is the code for this:
```
# Calculates Jaccard Similarity for Description/Definition #
def modified_jaccard_similarity(self,term, description):
    term = self.stem(self.preprocess(term)).split(" ")
    description = self.stem(self.lemmatize(self.preprocess(self.extract_NN_JJ(description)))).split(" ")
    intersection = list(set(term).intersection(description))
    return int((len(intersection)/len(term))*100)
```

For each synonym in each uid, the score is calculated: which is shown below:
```
# If term is only one word, just take the syn_score as its final score, otherwise take the max #
score = syn_score if len(term.split()) == 1 else max(syn_score,def_score)
```

Then after all the synonym/description scores are computed, the best one is selected and used to determine if the minimum threshold was met. If it's met, the counter variable is increased because a term score's met the threshold. This logic is shown here:
```
 # Gets the dictionary with the highest final score (description score + synonym score) and it's corresponding data #
 best_score_data = max(scores, key=lambda x: x['Final_Score'])
 
 # 'Holder' consists of the score that was generated to determine if the MINIMUM scoring_threshold was met #
 best_score = best_score_data['Holder']
 results.append(best_score_data)
                    
 # If best score is greater than or equal to the threshold, increase counter (a step closer to threshold) #
 if best_score >= self.score_threshold or threshold == 1:
    counter += 1
 
 # If threshold is met, then return results #
 if counter == threshold: return results
```
## Future Work
More ontologies can be incorporated into the API. It will require creating a new child class for that specific ontology. The scoring logic can be kept the same, with the exception of writing code to parse the data. A list of all ontologies supported with the NCBI e-utilities API can be found [here](https://www.ncbi.nlm.nih.gov/books/NBK25497/table/chapter2.T._entrez_unique_identifiers_ui/?report=objectonly).
