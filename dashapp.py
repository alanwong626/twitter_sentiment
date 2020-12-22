# set chdir to current dir
import os
import sys
sys.path.insert(0, os.path.realpath(os.path.dirname(__file__)))
os.chdir(os.path.realpath(os.path.dirname(__file__)))

import dash
from dash.dependencies import Output, Input, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc


import plotly
import plotly.graph_objs as go
import plotly.express as px
import sqlite3
import pandas as pd

from collections import Counter
import string
import regex as re
from cache import cache
from config import stop_words
import time
import pickle

import json
import requests
import xmltojson
import numpy as np
from textblob import TextBlob
from datetime import datetime


#############
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import matplotlib

matplotlib.use('agg')

def fig_to_uri(in_fig, close_all=True, **save_args):
    # type: (plt.Figure) -> str
    """
    Save a figure as a URI
    :param in_fig:
    :return:
    """
    out_img = BytesIO()
    in_fig.savefig(out_img, format='png', transparent=True, **save_args)
    if close_all:
        in_fig.clf()
        plt.close('all')
    out_img.seek(0)  # rewind file
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    return "data:image/png;base64,{}".format(encoded)
###############

# it's ok to use one shared sqlite connection
# as we are making selects only, no need for any kind of serialization as well
conn = sqlite3.connect('twitter.db', check_same_thread=False)

punctuation = [str(i) for i in string.punctuation]



sentiment_colors = {-1:"#EE6055",
                    -0.5:"#FDE74C",
                     0:"#FFE6AC",
                     0.5:"#D0F2DF",
                     1:"#9CEC5B",}


# sentiment_colors = {-1:"#DC143C",
#                     -0.5:"#FD8080",
#                      0:"#FFE6AC",
#                      0.5:"#DCF8C6",
#                      1:"#25D366",}


app_colors = {
    # 'background': '#0C0F0A',
    'background': '#1A1A1D',
    'text': '#FFFFFF',
    'sentiment-plot':'#3500D3',
    'volume-bar':'#FC4445',
    'someothercolor':'#FF206E',
}

POS_NEG_NEUT = 0.1

MAX_DF_LENGTH = 100

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Sentiment Analysis"

###### Tab1 #######
tab1_content = html.Div(
    [   html.Div(className='container-fluid', children=[html.H2('Twitter Sentiment', style={'color':"#FFFFFF"}),
                                                        html.H5('Search:', style={'color':app_colors['text']}),
                                                  dcc.Input(id='sentiment_term', value='twitter', type='text', style={'color':app_colors['someothercolor'],'width':'78%'}),
                                                #   html.Button('Submit', id='submit-val', n_clicks=0, style={'width':'10%'}),
                                                  ],
                 style={'width':'98%','margin-left':10,'margin-right':10,'max-width':50000}),

        
        
        html.Div(className='row', children=[html.Div(id='related-sentiment', children=html.Button('Loading related terms...', id='related_term_button'), className='col s12 m6 l6', style={"word-wrap":"break-word"}),
                                            html.Div(id='recent-trending', className='col s12 m6 l6', style={"word-wrap":"break-word"})]),

        html.Div(className='row', children=[html.Div(dcc.Graph(id='live-graph', animate=False), className='col s12 m6 l6'),
                                            html.Div(dcc.Graph(id='historical-graph', animate=False), className='col s12 m6 l6')]),

        html.Div(className='row', children=[html.Div(id="recent-tweets-table", className='col s12 m6 l6'),
                                            html.Div(dcc.Graph(id='sentiment-pie', animate=False), className='col s12 m6 l6'),]),
        
        dcc.Interval(
            id='graph-update',
            interval=1*1000,
            n_intervals=0 
        ),
        dcc.Interval(
            id='historical-update',
            interval=60*1000,
            n_intervals=0 
        ),

        dcc.Interval(
            id='related-update',
            interval=30*1000,
            n_intervals=0 
        ),

        dcc.Interval(
            id='recent-table-update',
            interval=2*1000,
            n_intervals=0 
        ),

        dcc.Interval(
            id='sentiment-pie-update',
            interval=60*1000,
            n_intervals=0 
        ),

    ], style={'backgroundColor': app_colors['background'], 'margin-top':'-30px', 'height':'2000px', 'padding-right':'5%','padding-left':'5%', 'padding-right':'5%'},
)

tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderTop': '1px solid #1A1A1D',
    'borderBottom': '1px solid #1A1A1D',
    'borderLeft': '1px solid #1A1A1D',
    'borderRight': '1px solid #1A1A1D',    
    'padding': '6px',
    'fontWeight': 'bold',
    'backgroundColor': app_colors['background']
}
tab_selected_style = {
    'borderTop': '1px solid #1A1A1D',
    'borderBottom': '1px solid #1A1A1D',
    'borderLeft': '1px solid #1A1A1D',
    'borderRight': '1px solid #1A1A1D',  
    'backgroundColor': app_colors['background'],
    'color': 'white',
    'padding': '6px'
}

tab1 = dcc.Tab(tab1_content, id='tab-1', style=tab_style, selected_style=tab_selected_style)


###### Tab2 ####
# tab2_content = html.Div([
#     html.Div(className='container-fluid', children=[html.H2('RSS Sentiment', style={'color':"#FFFFFF"}),
#                                                         html.H5('Search:', style={'color':app_colors['text']}),
#                                                   dcc.Input(id='rss_term', value='twitter', type='text', style={'color':app_colors['someothercolor'],'width':'78%'}),
#                                                 #   html.Button('Submit', id='submit-val', n_clicks=0, style={'width':'10%'}),
#                                                   ],
#                  style={'width':'98%','margin-left':10,'margin-right':10,'max-width':50000}),

#     # dbc.Row([
#     #     dbc.Col()
#     # ])
################################################################################################################################################################################################
def bulletgraph(data=None, limits=None, labels=None, axis_label=None, title=None,
                size=(5, 3), palette=None, formatter=None, target_color="gray",
                bar_color="black", label_color="gray"):
    """ Build out a bullet graph image
        Args:
            data = List of labels, measures and targets
            limits = list of range valules
            laabels = list of descriptions of the limit ranges
            axis_label = string describing x axis
            title = string title of plot
            size = tuple for plot size
            palette = a seaborn palette
            formatter = matplotlib formatter object for x axis
            target_color = color string for the target line
            bar_color = color string for the small bar
            label_color = color string for the limit label text
        Returns:
            a matplotlib figure
    """
    # Determine the max value for adjusting the bar height
    # Dividing by 10 seems to work pretty well
    h = limits[-1] / 10

    # Use the green palette as a sensible default
    if palette is None:
#         palette = sns.light_palette("green", len(limits), reverse=False)
        palette = sns.color_palette("RdYlGn", len(limits))

    # Must be able to handle one or many data sets via multiple subplots
    if len(data) == 1:
        fig, ax = plt.subplots(figsize=size, sharex=True)
    else:
        fig, axarr = plt.subplots(len(data), figsize=size, sharex=True)

    # Add each bullet graph bar to a subplot
    for idx, item in enumerate(data):

        # Get the axis from the array of axes returned when the plot is created
        if len(data) > 1:
            ax = axarr[idx]

        # Formatting to get rid of extra marking clutter
        ax.set_aspect('equal')
        ax.set_yticklabels([item[0]])
        ax.set_yticks([1])
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        prev_limit = 0
        for idx2, lim in enumerate(limits):
            # Draw the bar
            ax.barh([1], lim - prev_limit, left=prev_limit, height=h,
                    color=palette[idx2])
            prev_limit = lim
        rects = ax.patches
        # The last item in the list is the value we're measuring
        # Draw the value we're measuring
        ax.barh([1], item[1], height=(h / 3), color=bar_color)

        # Need the ymin and max in order to make sure the target marker
        # fits
        ymin, ymax = ax.get_ylim()
        ax.vlines(
            item[2], ymin * .9, ymax * .9, linewidth=1.5, color=target_color)

    # Now make some labels
    if labels is not None:
        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                -height * .4,
                label,
                ha='center',
                va='bottom',
                color=label_color)
    if formatter:
        ax.xaxis.set_major_formatter(formatter)
    if axis_label:
        ax.set_xlabel(axis_label)
    if title:
        fig.suptitle(title, fontsize=14)
    fig.subplots_adjust(hspace=0)
    return fig
    
def plot_bullet(value,title = "", x_label = "",y_label=""):
    data_to_plot2 = [("{}".format(title),value , 100)]
    sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})
    fig = bulletgraph(data_to_plot2, limits=[25, 50, 75, 100],
                labels=["Negative", " ", "  ", "Positive"], size=(5,6),
                axis_label="{}".format(x_label), label_color="black",
                bar_color="#252525", target_color='#f7f7f7',
                title="{}".format(y_label))
    fig.set_canvas(plt.gcf().canvas)
    return fig

################################################################################################################################################################################################################################

def get_news(query,days=1):
    def google_news(query, days):
        link = "https://news.google.com/news/rss/headlines/section/topic/BUSINESS/search?q={}+when:{}d".format(query,days)
        return link
    link = google_news(query,days)
    r = requests.get(link)
    res = json.loads(xmltojson.parse(requests.get(link).text))
    res['rss']["channel"]['item']

    headlines =[]
    for item in res['rss']["channel"]['item']:
        headline = {}
        headline['Date'] = item['pubDate']
        headline['Title'] = item['title']
        headline['Link'] = item["link"]
        headlines.append(headline)
    news = pd.DataFrame(headlines)
    polarity = lambda x: round(TextBlob(x).sentiment.polarity,2)
    subjectivity = lambda x: round(TextBlob(x).sentiment.subjectivity,2)
    news_polarity = np.zeros(len(news['Title']))
    news_subjectivity = np.zeros(len(news['Title']))
    for idx, headline in enumerate(news["Title"]):
    #     try:
        news_polarity[idx] = polarity(headline)
        news_subjectivity[idx] = subjectivity(headline)
    #     except:
    #         pass
    news["Polarity"]=news_polarity
    date = lambda x : datetime.strptime(x.split(",")[1][1:-4],'%d %b %Y %H:%M:%S')
    news['Date'] = news["Date"].apply(date)
    return news[:50]

tab2_content = html.Div([
    html.Div(className='container-fluid', children=[html.H2('RSS Sentiment', style={'color':"#FFFFFF"}),
                                                        html.H5('Search:', style={'color':app_colors['text']}),
                                                  dcc.Input(id='rss_term', value='twitter', type='text', style={'color':app_colors['someothercolor'],'width':'78%'}),
                                                  html.Button('Submit', id='rss', n_clicks=0, style={'width':'10%'}),
                                                  ],
                 style={'width':'98%','margin-left':10,'margin-right':10,'max-width':50000}),
    
    
    
    html.Div(className='row', children=[html.Div(id="recent-news-table", className='col s12 m6 l6'),
                                        html.Div([html.Img(id = 'sentiment_plot', src = '')],
                                                id='plot_div')
                                        # html.Div(dcc.Graph(id='news-pie', animate=False), className='col s12 m6 l6'),
                                        ]),
    
    dcc.Interval(
        id='news-table-update',
        interval=2*1000,
        n_intervals=0 
    ),

    dcc.Interval(
        id='news-pie-update',
        interval=60*1000,
        n_intervals=0 
    ),

    
],style={'backgroundColor': app_colors['background'], 'margin-top':'-30px', 'height':'2000px', 'padding-right':'5%','padding-left':'5%', 'padding-right':'5%'},)


@app.callback(Output('recent-news-table', 'children'),
              [Input('rss', 'n_clicks')],
              [State('rss_term', 'value')],
              prevernt_initial_callback = True)
                  
def update_recent_news(n_clicks, value):
    if n_clicks:
        if value:
            df = get_news(value)
        else:
            df = pd.DataFrame([])


        df = df[['Date','Title','Polarity']]

        return generate_table(df, max_rows=10)
    return None


# @app.callback(Output('news-pie', 'figure'),
#               [Input(component_id='rss_term', component_property='value'),
#               Input('news-pie-update','n_intervals')])
# def update_pie_chart(rss_term,n_intervals):


#     return px.line()

@app.callback(
    Output('sentiment_plot', 'src'),
    [Input('rss', 'n_clicks')],
    [State('rss_term', 'value')],
    prevent_initial_callback = True
)
def update_graph(n_clicks, value):
    if n_clicks:
        df = get_news(value)
        score = round((df["Polarity"].mean()+1)/(2)*100)
        fig = plot_bullet(score)
        # fig, ax1 = plt.subplots(1,1)
        # np.random.seed(len(input_value))
        # ax1.matshow(np.random.uniform(-1,1, size = (n_val,n_val)))
        # ax1.set_title(input_value)
        out_url = fig_to_uri(fig)
        return out_url
    return None


###################################################################################
    


# ],style={'backgroundColor': app_colors['background'], 'margin-top':'-30px', 'height':'2000px', 'padding-right':'5%','padding-left':'5%', 'padding-right':'5%'})

tab2 = dcc.Tab(tab2_content,id='tab-2', style=tab_style, selected_style=tab_selected_style)



app.layout = dcc.Tabs([
    tab1,
    tab2
])


def df_resample_sizes(df, maxlen=MAX_DF_LENGTH):
    df_len = len(df)
    resample_amt = 100
    vol_df = df.copy()
    vol_df['volume'] = 1

    ms_span = (df.index[-1] - df.index[0]).seconds * 1000
    rs = int(ms_span / maxlen)

    df = df.resample('{}ms'.format(int(rs))).mean()
    df.dropna(inplace=True)

    vol_df = vol_df.resample('{}ms'.format(int(rs))).sum()
    vol_df.dropna(inplace=True)

    df = df.join(vol_df['volume'])

    return df

# make a counter with blacklist words and empty word with some big value - we'll use it later to filter counter
stop_words.append('')
blacklist_counter = Counter(dict(zip(stop_words, [1000000]*len(stop_words))))

# complie a regex for split operations (punctuation list, plus space and new line)
split_regex = re.compile("[ \n"+re.escape("".join(punctuation))+']')

def related_sentiments(df, sentiment_term, how_many=15):
    try:

        related_words = {}

        # it's way faster to join strings to one string then use regex split using your punctuation list plus space and new line chars
        # regex precomiled above
        tokens = split_regex.split(' '.join(df['tweet'].values.tolist()).lower())

        # it is way faster to remove stop_words, sentiment_term and empty token by making another counter
        # with some big value and substracting (counter will substract and remove tokens with negative count)
        blacklist_counter_with_term = blacklist_counter.copy()
        blacklist_counter_with_term[sentiment_term] = 1000000
        counts = (Counter(tokens) - blacklist_counter_with_term).most_common(15)

        for term,count in counts:
            try:
                df = pd.read_sql("SELECT sentiment.* FROM  sentiment_fts fts LEFT JOIN sentiment ON fts.rowid = sentiment.id WHERE fts.sentiment_fts MATCH ? ORDER BY fts.rowid DESC LIMIT 200", conn, params=(term,))
                related_words[term] = [df['sentiment'].mean(), count]
            except Exception as e:
                with open('errors.txt','a') as f:
                    f.write(str(e))
                    f.write('\n')

        return related_words

    except Exception as e:
        with open('errors.txt','a') as f:
            f.write(str(e))
            f.write('\n')




# def quick_color(s):
#     # except return bg as app_colors['background']
#     if s >= POS_NEG_NEUT:
#         # positive
#         return "#002C0D"
#     elif s <= -POS_NEG_NEUT:
#         # negative:
#         return "#270000"

#     else:
#         return app_colors['background']


# app_colors = {
#     # 'background': '#0C0F0A',
#     'background': '#1A1A1D',
#     'text': '#FFFFFF',
#     'sentiment-plot':'#3500D3',
#     'volume-bar':'#FC4445',
#     'someothercolor':'#FF206E',
# }
        
def quick_color(s):
    # except return bg as app_colors['background']
    if s >= POS_NEG_NEUT:
        # positive
        return "#3500D3"
    elif s <= -POS_NEG_NEUT:
        # negative:
        return "#FC4445"

    else:
        return app_colors['background']

def generate_table(df, max_rows=10):
    return html.Table(className="responsive-table",
                      children=[
                          html.Thead(
                              html.Tr(
                                  children=[
                                      html.Th(col.title()) for col in df.columns.values],
                                  style={'color':app_colors['text']}
                                  )
                              ),
                          html.Tbody(
                              [
                                  
                              html.Tr(
                                  children=[
                                      html.Td(data) for data in d
                                      ], style={'color':app_colors['text'],
                                                'background-color':quick_color(d[2])}
                                  )
                               for d in df.values.tolist()])
                          ]
    )


def pos_neg_neutral(col):
    if col >= POS_NEG_NEUT:
        # positive
        return 1
    elif col <= -POS_NEG_NEUT:
        # negative:
        return -1

    else:
        return 0
    
            
@app.callback(Output('recent-tweets-table', 'children'),
              [Input('sentiment_term', 'value'),
              Input('recent-table-update','n_intervals')])        
def update_recent_tweets(sentiment_term, n_intervals):
    if sentiment_term:
        df = pd.read_sql("SELECT sentiment.* FROM sentiment_fts fts LEFT JOIN sentiment ON fts.rowid = sentiment.id WHERE fts.sentiment_fts MATCH ? ORDER BY fts.rowid DESC LIMIT 10", conn, params=(sentiment_term+'*',))
    else:
        df = pd.read_sql("SELECT * FROM sentiment ORDER BY id DESC, unix DESC LIMIT 10", conn)

    df['date'] = pd.to_datetime(df['unix'], unit='ms')

    df = df.drop(['unix','id'], axis=1)
    df = df[['date','tweet','sentiment']]

    return generate_table(df, max_rows=10)


@app.callback(Output('sentiment-pie', 'figure'),
              [Input(component_id='sentiment_term', component_property='value'),
              Input('sentiment-pie-update','n_intervals')])
def update_pie_chart(sentiment_term,n_intervals):

    # get data from cache
    for i in range(100):
        sentiment_pie_dict = cache.get('sentiment_shares', sentiment_term)
        if sentiment_pie_dict:
            break
        time.sleep(0.1)

    if not sentiment_pie_dict:
        return None

    labels = ['Positive','Negative']

    try: pos = sentiment_pie_dict[1]
    except: pos = 0

    try: neg = sentiment_pie_dict[-1]
    except: neg = 0

    
    
    values = [pos,neg]
    colors = ['#007F25', '#800000']

    trace = go.Pie(labels=labels, values=values,
                   hoverinfo='label+percent', textinfo='value', 
                   textfont=dict(size=20, color=app_colors['text']),
                   marker=dict(colors=colors, 
                               line=dict(color=app_colors['background'], width=2)))

    return {"data":[trace],'layout' : go.Layout(
                                                  title='Positive vs Negative sentiment for "{}" (longer-term)'.format(sentiment_term),
                                                  font={'color':app_colors['text']},
                                                  plot_bgcolor = app_colors['background'],
                                                  paper_bgcolor = app_colors['background'],
                                                  showlegend=True)}




@app.callback(Output('live-graph', 'figure'),
              [Input(component_id='sentiment_term', component_property='value'),
              Input('graph-update', 'n_intervals')])
def update_graph_scatter(sentiment_term,n_intervals):
    try:
        if sentiment_term:
            df = pd.read_sql("SELECT sentiment.* FROM sentiment_fts fts LEFT JOIN sentiment ON fts.rowid = sentiment.id WHERE fts.sentiment_fts MATCH ? ORDER BY fts.rowid DESC LIMIT 1000", conn, params=(sentiment_term+'*',))
        else:
            df = pd.read_sql("SELECT * FROM sentiment ORDER BY id DESC, unix DESC LIMIT 1000", conn)
        df.sort_values('unix', inplace=True)
        df['date'] = pd.to_datetime(df['unix'], unit='ms')
        df.set_index('date', inplace=True)
        init_length = len(df)
        df['sentiment_smoothed'] = df['sentiment'].rolling(int(len(df)/5)).mean()
        df = df_resample_sizes(df)
        X = df.index
        Y = df.sentiment_smoothed.values
        Y2 = df.volume.values
        data = plotly.graph_objs.Scatter(
                x=X,
                y=Y,
                name='Sentiment',
                mode= 'lines',
                yaxis='y2',
                line = dict(color = (app_colors['sentiment-plot']),
                            width = 4,)
                )

        data2 = plotly.graph_objs.Bar(
                x=X,
                y=Y2,
                name='Volume',
                marker=dict(color=app_colors['volume-bar']),
                )

        return {'data': [data,data2],'layout' : go.Layout(xaxis=dict(range=[min(X),max(X)]),
                                                          yaxis=dict(range=[min(Y2),max(Y2*4)], title='Volume', side='right'),
                                                          yaxis2=dict(range=[min(Y),max(Y)], side='left', overlaying='y',title='sentiment'),
                                                          title='Live sentiment for: "{}"'.format(sentiment_term),
                                                          font={'color':app_colors['text']},
                                                          plot_bgcolor = app_colors['background'],
                                                          paper_bgcolor = app_colors['background'],
                                                          showlegend=False)}

    except Exception as e:
        with open('errors.txt','a') as f:
            f.write(str(e))
            f.write('\n')

@app.callback(Output('historical-graph', 'figure'),
              [Input(component_id='sentiment_term', component_property='value'),
               Input('historical-update','n_intervals')])
def update_hist_graph_scatter(sentiment_term,n_intervals):
    try:
        if sentiment_term:
            df = pd.read_sql("SELECT sentiment.* FROM sentiment_fts fts LEFT JOIN sentiment ON fts.rowid = sentiment.id WHERE fts.sentiment_fts MATCH ? ORDER BY fts.rowid DESC LIMIT 10000", conn, params=(sentiment_term+'*',))
        else:
            df = pd.read_sql("SELECT * FROM sentiment ORDER BY id DESC, unix DESC LIMIT 10000", conn)
        df.sort_values('unix', inplace=True)
        df['date'] = pd.to_datetime(df['unix'], unit='ms')
        df.set_index('date', inplace=True)
        # save this to a file, then have another function that
        # updates because of this, using intervals to read the file.
        # https://community.plot.ly/t/multiple-outputs-from-single-input-with-one-callback/4970

        # store related sentiments in cache
        cache.set('related_terms', sentiment_term, related_sentiments(df, sentiment_term), 120)

        #print(related_sentiments(df,sentiment_term), sentiment_term)
        init_length = len(df)
        df['sentiment_smoothed'] = df['sentiment'].rolling(int(len(df)/5)).mean()
        df.dropna(inplace=True)
        df = df_resample_sizes(df,maxlen=500)
        X = df.index
        Y = df.sentiment_smoothed.values
        Y2 = df.volume.values

        data = plotly.graph_objs.Scatter(
                x=X,
                y=Y,
                name='Sentiment',
                mode= 'lines',
                yaxis='y2',
                line = dict(color = (app_colors['sentiment-plot']),
                            width = 4,)
                )

        data2 = plotly.graph_objs.Bar(
                x=X,
                y=Y2,
                name='Volume',
                marker=dict(color=app_colors['volume-bar']),
                )

        df['sentiment_shares'] = list(map(pos_neg_neutral, df['sentiment']))

        #sentiment_shares = dict(df['sentiment_shares'].value_counts())
        cache.set('sentiment_shares', sentiment_term, dict(df['sentiment_shares'].value_counts()), 120)

        return {'data': [data,data2],'layout' : go.Layout(xaxis=dict(range=[min(X),max(X)]), # add type='category to remove gaps'
                                                          yaxis=dict(range=[min(Y2),max(Y2*4)], title='Volume', side='right'),
                                                          yaxis2=dict(range=[min(Y),max(Y)], side='left', overlaying='y',title='sentiment'),
                                                          title='Longer-term sentiment for: "{}"'.format(sentiment_term),
                                                          font={'color':app_colors['text']},
                                                          plot_bgcolor = app_colors['background'],
                                                          paper_bgcolor = app_colors['background'],
                                                          showlegend=False)}

    except Exception as e:
        with open('errors.txt','a') as f:
            f.write(str(e))
            f.write('\n')



max_size_change = .4

def generate_size(value, smin, smax):
    size_change = round((( (value-smin) /smax)*2) - 1,2)
    final_size = (size_change*max_size_change) + 1
    return final_size*120
    
    


# SINCE A SINGLE FUNCTION CANNOT UPDATE MULTIPLE OUTPUTS...
#https://community.plot.ly/t/multiple-outputs-from-single-input-with-one-callback/4970

@app.callback(Output('related-sentiment', 'children'),
              [Input(component_id='sentiment_term', component_property='value'),
              Input('related-update','n_intervals')])

def update_related_terms(sentiment_term,n_intervals):
    try:

        # get data from cache
        for i in range(100):
            related_terms = cache.get('related_terms', sentiment_term) # term: {mean sentiment, count}
            if related_terms:
                break
            time.sleep(0.1)

        if not related_terms:
            return None

        buttons = [html.Button('{}({})'.format(term, related_terms[term][1]), id='related_term_button', value=term, className='btn', type='submit', style={'background-color':'#4CBFE1',
                                                                                                                                                           'margin-right':'5px',
                                                                                                                                                           'margin-top':'5px'}) for term in related_terms]
        #size: related_terms[term][1], sentiment related_terms[term][0]
        

        sizes = [related_terms[term][1] for term in related_terms]
        smin = min(sizes)
        smax = max(sizes) - smin  

        buttons = [html.H5('Terms related to "{}": '.format(sentiment_term), style={'color':app_colors['text']})]+[html.Span(term, style={'color':sentiment_colors[round(related_terms[term][0]*2)/2],
                                                              'margin-right':'15px',
                                                              'margin-top':'15px',
                                                              'font-size':'{}%'.format(generate_size(related_terms[term][1], smin, smax))}) for term in related_terms]


        return buttons
        

    except Exception as e:
        with open('errors.txt','a') as f:
            f.write(str(e))
            f.write('\n')


#recent-trending div
# term: [sent, size]

@app.callback(Output('recent-trending', 'children'),
              [Input(component_id='sentiment_term', component_property='value'),
              Input('related-update','n_intervals')])
def update_recent_trending(sentiment_term,n_intervals):
    try:
        query = """
                SELECT
                        value
                FROM
                        misc
                WHERE
                        key = 'trending'
        """

        c = conn.cursor()

        result = c.execute(query).fetchone()

        related_terms = pickle.loads(result[0])



##        buttons = [html.Button('{}({})'.format(term, related_terms[term][1]), id='related_term_button', value=term, className='btn', type='submit', style={'background-color':'#4CBFE1',
##                                                                                                                                                           'margin-right':'5px',
##                                                                                                                                                           'margin-top':'5px'}) for term in related_terms]
        #size: related_terms[term][1], sentiment related_terms[term][0]
        

        sizes = [related_terms[term][1] for term in related_terms]
        smin = min(sizes)
        smax = max(sizes) - smin  

        buttons = [html.H5('Recently Trending Terms: ', style={'color':app_colors['text']})]+[html.Span(term, style={'color':sentiment_colors[round(related_terms[term][0]*2)/2],
                                                              'margin-right':'15px',
                                                              'margin-top':'15px',
                                                              'font-size':'{}%'.format(generate_size(related_terms[term][1], smin, smax))}) for term in related_terms]


        return buttons
        

    except Exception as e:
        with open('errors.txt','a') as f:
            f.write(str(e))
            f.write('\n')


            

server = app.server
dev_server = app.run_server
