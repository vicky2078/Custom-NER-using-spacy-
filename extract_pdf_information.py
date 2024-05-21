import fitz
import re
import spacy 
nlp = spacy.load('en_core_web_lg') 
import stringcase
import json
import glob
import os



nlp_ner = spacy.load("model/model-best-7820")



mypath = "sapcy-ner/text_pre/doc"

o=[]
for file in glob.glob(mypath + "/*.pdf"):
    file_name = os.path.basename(file)
    file_name1=os.path.splitext(file_name)[0]  
    fname=file
    
    with fitz.open(fname) as doc:
        text = chr(12).join([page.get_text(sort=True) for page in doc])
      
    sentence  = ' '.join(text.split())   
    
    complete_doc = nlp(sentence)
    def is_token_allowed(token):
        return bool(token
                    and str(token).strip()
                    and not token.is_stop
                    and not token.is_punct
                   )
    
    def preprocess_token(token):
        return token.lemma_.strip().lower()    
    
    
    complete_filtered_tokens = [preprocess_token(token)for token in complete_doc if is_token_allowed(token)] 

    out=' '.join(map(str,complete_filtered_tokens))

    punctuation_remove = re.sub(r'[^\w\s]', '', out)
    
    
    doc=nlp_ner(punctuation_remove)
    out=[(ent.label_, ent.text) for ent in doc.ents]
    result = dict(out)

    print(result)




   