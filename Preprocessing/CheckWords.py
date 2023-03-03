import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
df = pd.read_csv(r'BioNLP2023-1A-Train.csv')
#remove rows with empty GT
df = df.dropna(subset=['Summary']).reset_index(drop=True)

df = df.replace(to_replace ='\[\*\*(.*?)\*\*\]', value = '', regex = True)

total_percent = 0
total_matches = 0
total_sum = 0
for idx in range(len(df.index)):
    #remove image id
    summary = str(df.loc[idx]['Summary'])
    text = str(df.loc[idx]['Subjective Sections']) + ' ' + str(df.loc[idx]['Objective Sections']) + ' ' + str(df.loc[idx]['Assessment'])
    sum_w = re.findall(r'\w+', summary.lower())
    text_w = re.findall(r'\w+', text.lower())
    for w in sum_w:
        if w in stop_words:
            sum_w.remove(w)
    for w in text_w:
        if w in stop_words:
            text_w.remove(w)

    total_s = len(sum_w)
    matches = 0
    for word in sum_w:
        if word in text_w:
            matches+=1
    percent = matches/total_s
    print(str(matches) + '/' + str(total_s) + '=' + str(percent))
    total_matches += matches
    total_sum += total_s
    total_percent += percent

print(str(total_matches) + '/' + str(total_sum) + ' Percent ' + str(total_matches/total_sum))
print('Average ' + str(total_percent/len(df.index)))
