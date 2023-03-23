import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import medialpy

# python 3.6
#from nlpre import replace_acronyms
#pairs = replace_acronyms('h/o hyperglycemia chronic obstructive pulmonary disease (copd, bronchitis, emphysema) with acute exacerbation a 59 year-old man presents with malaise and hypoxia')
#print(pairs)

df = pd.read_csv(r'BioNLP2023-1A-Train.csv')
#remove rows with empty GT
df = df.dropna(subset=['Summary']).reset_index(drop=True)

i_a = 0
i_a_2 = 0
i_s = 0
i_s_2 = 0
for idx in range(len(df.index)):
    #remove image id
    ass = str(df.loc[idx]['Assessment'])
    sub = str(df.loc[idx]['Subjective Sections'])
    words_a = word_tokenize(ass)
    print(ass)
    print(words_a)
    input()
    words_s = word_tokenize(sub)
    for i, w in enumerate(words_a):
        if medialpy.exists(w):
            i_a += 1
            term = medialpy.find(w)
            print(w + ': ')
            print(term.meaning)
            if len(term.meaning) == 1:
                words_a[i] = term.meaning[0]
            else:
                i_a_2 += 1
    input()
    print(sub)
    print(words_s)
    input()
    for i, w in enumerate(words_s):
        if medialpy.exists(w):
            i_s += 1
            term = medialpy.find(w)
            print(w + ': ')
            print(term.meaning)
            if len(term.meaning) == 1:
                words_s[i] = term.meaning[0]
            else:
                i_s_2 += 1
    input()

print(i_a)
print(i_a_2)
print(i_s)
print(i_s_2)

#df.to_csv(r'BioNLP_Abbr.csv', index=False)