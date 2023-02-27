import PySimpleGUI as sg
from ast import literal_eval
import pandas as pd

# Initial constants and vars
filename = ''
tags = [[],[],[]]
tokens = [[],[],[]]
idx = 0
# Color for B-I-O (2-3-0) Label following BC5CDR dataset Tag dictionary
colors = {0: 'grey', 2: 'red', 3:'green'}


# NER checking layouts
def make_next_window():
    print('Window ' + str(idx))
    # Making list of buttons for every token with the color of the tag
    text_a = [[sg.Button(str(word), pad=(0,0), button_color=colors[tags[0][idx][i]], border_width=0, key=(idx,0,i)) for i, word in enumerate(tokens[0][idx])]]
    text_s = [[sg.Button(str(word), pad=(0,0), button_color=colors[tags[1][idx][i]], border_width=0, key=(idx,1,i)) for i, word in enumerate(tokens[1][idx])]]
    text_o = [[sg.Button(str(word), pad=(0,0), button_color=colors[tags[2][idx][i]], border_width=0, key=(idx,2,i)) for i, word in enumerate(tokens[2][idx])]]

    layout = [[sg.Text('Index: ' + str(idx) + '/' + str(len(df.index)), key='_INDEX_')],
            [sg.Column([[sg.Frame('Assessment', text_a, pad=10, key='_TEXT_A_')],
                        [sg.Frame('Subjective', text_s, pad=10, key='_TEXT_S_')],
                        [sg.Frame('Objective', text_o, pad=10, key='_TEXT_O_')]],
                    key='_TEXT_',scrollable=True, expand_y=True)],
            [sg.Text('Summary: ' + df.loc[idx]['Summary'])],
            [sg.Button('Next'), sg.Button('Save') ,sg.Exit()]]
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
        df.to_csv(filename)
        print('File saved = ' + filename)
    else:
        # Toggle BIO tag
        if tags[event[1]][idx][event[2]] == 0:
            tags[event[1]][idx][event[2]] = 2
        elif tags[event[1]][idx][event[2]] == 2:
            tags[event[1]][idx][event[2]] = 3
        elif tags[event[1]][idx][event[2]] == 3:
            tags[event[1]][idx][event[2]] = 0
        # Updating color on layout
        window2[event].update(tokens[event[1]][idx][event[2]], button_color=colors[tags[event[1]][idx][event[2]]])
    
window.close()