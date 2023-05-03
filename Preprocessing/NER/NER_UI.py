#
# UI Script to review and modify NER labeling
# 
#
# In order to be able to run the script the following packages:
#  pip install pysimplegui
#  brew install python-tk
#
# Once started the UI select the csv file, select the example number and press LAUCH.
# In the window for modifying the tags for each token, to change the tag click on the token
# and the color will change looping for the tags de 0(grey) - B(red) - I(green) in this order.
# To move between the examples use the NEXT & PREVIOUS buttons, and to save the changes click SAVE.
#
import PySimpleGUI as sg
from ast import literal_eval
import pandas as pd

# Initial constants and vars
filename = ''
tags = [[],[],[]]
tokens = [[],[],[]]
refs = [[],[],[]]
idx = 0
reference = 1
# Color for B-I-O (1-2-0) Label following BC5CDR dataset Tag dictionary (reworked for only disease)
colors = {0: 'grey', 1: 'red', 2:'green'}


# NER checking layouts
def make_next_window():
    print('Window ' + str(idx))
    # Making list of buttons for every token with the color of the tag
    text_a = [[sg.Button(str(word), pad=(0,0), button_color=colors[tags[0][idx][i]], border_width=0, key=(idx,0,i)) for i, word in enumerate(tokens[0][idx])]]
    text_s = [[sg.Button(str(word), pad=(0,0), button_color=colors[tags[1][idx][i]], border_width=0, key=(idx,1,i)) for i, word in enumerate(tokens[1][idx])]]
    text_o = [[sg.Button(str(word), pad=(0,0), button_color=colors[tags[2][idx][i]], border_width=0, key=(idx,2,i)) for i, word in enumerate(tokens[2][idx])]]

    if reference == 1:
        layout = [[sg.Text('Index: ' + str(idx) + '/' + str(len(df.index)), key='_INDEX_')],
                [sg.Column([[sg.Text(refs[0][idx])],
                            [sg.Frame('Assessment', text_a, pad=10, key='_TEXT_A_')],
                            [sg.Text(refs[1][idx])],
                            [sg.Frame('Subjective', text_s, pad=10, key='_TEXT_S_')],
                            [sg.Text(refs[2][idx])],
                            [sg.Frame('Objective', text_o, pad=10, key='_TEXT_O_')]],
                        key='_TEXT_',scrollable=True, expand_y=True)],
                [sg.Column([[sg.Text('Summary: ' + df.loc[idx]['Summary'])]],key='_TEXT_',scrollable=True)],
                [sg.Button('Previous'), sg.Button('Next'), sg.Button('Save') ,sg.Exit()]]
    else:
        layout = [[sg.Text('Index: ' + str(idx) + '/' + str(len(df.index)), key='_INDEX_')],
                [sg.Column([[sg.Frame('Assessment', text_a, pad=10, key='_TEXT_A_')],
                            [sg.Frame('Subjective', text_s, pad=10, key='_TEXT_S_')],
                            [sg.Frame('Objective', text_o, pad=10, key='_TEXT_O_')]],
                        key='_TEXT_',scrollable=True, expand_y=True)],
                [sg.Column([[sg.Text('Summary: ' + df.loc[idx]['Summary'])]],key='_TEXT_',scrollable=True)],
                [sg.Button('Previous'), sg.Button('Next'), sg.Button('Save') ,sg.Exit()]]
    return sg.Window('Example ' + str(idx), layout, size=(1600,600), finalize=True, resizable=True)


# Starting layout browse file and select starting example
layout1 = [[sg.Text('CSV File', size=(15, 1)), sg.InputText(key='-FILENAME-'), sg.FileBrowse()],
            [sg.Text('Select starting Example Number')],
            [sg.Input(default_text='0', key='_IN_')],
            [sg.Button('Launch'), sg.Exit()]]

window = sg.Window('NER', layout1, size=(1600,600),resizable=True)
window2 = None
w=1
while True:
    # Selecting window attention
    if w == 1:
        event, values = window.read()
    else:
        event, values = window2.read()
    print(event, values)

    # Managing button events
    if event == sg.WIN_CLOSED or event == 'Exit':
        if w == 1:
            break
        else:
            window2.close()
            window2 = None
            window.un_hide()
            w = 1
    elif event == 'Next':
        idx = idx+1
        window2.close()
        window2 = None
        window2 = make_next_window()
    elif event == 'Previous':
        idx = idx-1
        window2.close()
        window2 = None
        window2 = make_next_window()
    elif event == 'Launch':
        print('File selected = ' + values['-FILENAME-'])
        filename = values['-FILENAME-']
        # Loading csv NER tokens and tags
        df = pd.read_csv(filename)
        for i in range(len(df.index)):
            tags[0].append(literal_eval(df.loc[i]['tags_ap']))
            tags[1].append(literal_eval(df.loc[i]['tags_s']))
            tags[2].append(literal_eval(df.loc[i]['tags_o']))
            tokens[0].append(literal_eval(df.loc[i]['tokens_ap']))
            tokens[1].append(literal_eval(df.loc[i]['tokens_s']))
            tokens[2].append(literal_eval(df.loc[i]['tokens_o']))
            refs[0].append(df.loc[i]['Assessment'])
            refs[1].append(df.loc[i]['Subjective Sections'])
            refs[2].append(df.loc[i]['Objective Sections'])
        # Changing window
        window.hide()
        idx = int(values['_IN_'])
        window2 = make_next_window()
        w = 2
    elif event == 'Save':
        df['tokens_ap'] = tokens[0]
        df['tags_ap'] = tags[0]
        df['tokens_s'] = tokens[1]
        df['tags_s'] = tags[1]
        df['tokens_o'] = tokens[2]
        df['tags_o'] = tags[2]
        df.to_csv(filename, index=False)
        print('File saved = ' + filename)
    else:
        # Toggle BIO tag
        if tags[event[1]][idx][event[2]] == 0:
            tags[event[1]][idx][event[2]] = 1
        elif tags[event[1]][idx][event[2]] == 1:
            tags[event[1]][idx][event[2]] = 2
        elif tags[event[1]][idx][event[2]] == 2:
            tags[event[1]][idx][event[2]] = 0
        # Updating color on layout
        window2[event].update(tokens[event[1]][idx][event[2]], button_color=colors[tags[event[1]][idx][event[2]]])
    
window.close()