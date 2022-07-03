import re
import joblib
from nltk.tokenize import word_tokenize,MWETokenizer

#load model
crf = joblib.load('crf_ner_model.pkl')

#features for all words
def word2features(sent, i):
    word = sent[i][0]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),## is_first_capital
        'word.istitle()': word.istitle(),## Check if each word start with an upper case letter
        'word.isdigit()': word.isdigit(),## is_numeric
        'word.position()': str(i),
    }
    # Features for words that are not at the beginning of a document
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:word.position()': str(i),
        })
    else:
        features['BOS'] = True

    # Features for words that are not at the end of a document
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:word.position()': str(i),
        })
    else:
        # Indicate that it is the 'end of a document'
        features['EOS'] = True

    return features

# functions for extracting features in documents
def extract_features(doc):
    return [word2features(doc, i) for i in range(len(doc))]

# A function fo generating the list of labels for each document
def get_labels(doc):
    return [label for (token, postag, label) in doc]

def number_rule(number):
    if re.search(r'(?:(\d+\/\d+)|(\d+\/\d+\/\d+))',number):
        return True
    else:
        return False

def have_number(text):
    if len(text) - len(re.sub('\d+', '', text)) > 0:
        return True 
    return False

def get_map_entity(pred):
    project = []
    number = []
    alley =[]
    lane = []
    hamlet=[]
    to = []
    street = []
    ward = []
    dist = []
    city = []
    map = {}

    try:
        if len(pred) == 0:
            number.append(pred[0])
        if pred[0][1].endswith("OTHER") and number_rule(pred[0][0]) and (pred[1][1] == 'B_STREET' or pred[1][1] == 'STREET_TYPE' or pred[1][1] == 'OTHER'):
            number.append(pred[0][0].lower())

        for i in range(len(pred)):
            if pred[i][1].endswith("PRO"):
                project.append(pred[i][0].lower())
            if pred[i][1].endswith("NUMBER") and have_number(pred[i][0]): ## được xác định là number & chứa kí tự số
                number.append(pred[i][0].lower())
            if pred[i][1].endswith("ALLEY"):
                alley.append(pred[i][0].lower())
            if pred[i][1].endswith("LANE"):
                lane.append(pred[i][0].lower())
            if pred[i][1].endswith("TO"):
                to.append(pred[i][0].lower())
            if pred[i][1].endswith("HAMLET"):
                hamlet.append(pred[i][0].lower())
            if pred[i][1].endswith("STREET"):
                street.append(pred[i][0].lower())
            if pred[i][1].endswith("WARD"):
                ward.append(pred[i][0].lower())
            if pred[i][1].endswith("DIST"):
                dist.append(pred[i][0].lower())
            if pred[i][1].endswith("CITY"):
                city.append(pred[i][0].lower())
        
        map["project"] = " ".join(project)
        map["number"] = " ".join(number)
        map["alley"] = " ".join(alley)
        map["lane"] = " ".join(lane)
        map["to"] = " ".join(to)
        map["hamlet"] = " ".join(hamlet)
        map["street"] = " ".join(street)
        map["ward"] = " ".join(ward)
        map["dist"] = " ".join(dist)
        map["city"] = " ".join(city)
    except:
        print("pred -------------")
        print(pred)
        print("end -------------")
    return map

word_tokenizer = MWETokenizer(separator='')
def prepare_text(text):
    text = word_tokenizer.tokenize(word_tokenize(text)) 
    return ' '.join(text)
    
def extract_entity(address):
    temp = []
    for i in address.split(" "):
        if "," not in i and "." not in i:
            temp.append(i)
        if "," in i :
            temp.append(",")
        if "." in i :
            temp.append(".")
    return temp

def detect_entity(address):
    text = prepare_text(address)
    temp = extract_entity(text)
    detect = []
    for j in temp:
        l = []
        l.append(j)
        l.append("O")
        detect.append(tuple(l))
    arr = []
    arr.append(detect)
    X_detect = [extract_features(s) for s in arr]
    #open model
    y_detect = crf.predict(X_detect)
    pred = []
    for i in range(len(temp)):
        k = temp[i]
        v = y_detect[0][i]
        kv = []
        kv.append(k)
        kv.append(v)
        pred.append(tuple(kv))
    return get_map_entity(pred)

if __name__ == "__main__":
    address = "Số Nhà 29a,Ngách 12, ấp Tam Trinh   Ngõ 136 Đường Tam Trinh,Tổ 16"
    print(detect_entity(address))
