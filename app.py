import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import warnings
import sys
import re
import csv
import math
import random
import timeit

import urllib
import numpy as np
import statsmodels.api as sm
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.stats.outliers_influence import summary_table


px.set_mapbox_access_token('pk.eyJ1Ijoia2xvY2V5IiwiYSI6ImNrYm9uaWhoYjI0ZDcycW56ZWExODRmYzcifQ.Mb27BYst186G4r5fjju6Pw')

#########################################################################################
################################# CONFIG APP ############################################
#########################################################################################

FONT_AWESOME = "https://use.fontawesome.com/releases/v5.10.2/css/all.css"

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

external_stylesheets=[dbc.themes.BOOTSTRAP, FONT_AWESOME, 'https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions = True
server = app.server

################################# LOAD DATA ##################################

gendat_df = pd.read_pickle('dataframe_data/GenDat4App_p4.pkl')
gendat_df[('S2_1_C2_2', 'Hospital State', '', 'Hospital State (S2_1_C2_2)')] = gendat_df[('S2_1_C2_2', 'Hospital State', '', 'Hospital State (S2_1_C2_2)')].replace(np.nan, 'Not given')
gendat_df[('Hospital type, text', 'Hospital type, text', 'Hospital type, text', 'Hospital type, text')] = gendat_df[('Hospital type, text', 'Hospital type, text', 'Hospital type, text', 'Hospital type, text')].replace(np.nan, 'Not given')
gendat_df[('Control type, text', 'Control type, text', 'Control type, text', 'Control type, text')] = gendat_df[('Control type, text', 'Control type, text', 'Control type, text', 'Control type, text')].replace(np.nan, 'Not given')

######################## SELECTION LISTS #####################################

HOSPITALS = gendat_df[('Curated Name and Num', 'Curated Name and Num', 'Curated Name and Num', 'Curated Name and Num')].tolist()
CMS_NUMS = len(gendat_df[('PRVDR_NUM', 'Hospital Provider Number', 'HOSPITAL IDENTIFICATION INFORMATION', 'Hospital Provider Number (PRVDR_NUM)')].unique()) 
# 
beds = gendat_df[('S3_1_C2_27', 'Total Facility', 'NUMBER OF BEDS', 'Total Facility (S3_1_C2_27)')].tolist()

states = gendat_df[('S2_1_C2_2', 'Hospital State', '', 'Hospital State (S2_1_C2_2)')].tolist()
htypes = gendat_df[('Hospital type, text', 'Hospital type, text', 'Hospital type, text', 'Hospital type, text')].tolist()
ctypes = gendat_df[('Control type, text', 'Control type, text', 'Control type, text', 'Control type, text')].tolist()

states = ['NaN' if x is np.nan else x for x in states]
htypes = ['NaN' if x is np.nan else x for x in htypes]
ctypes = ['NaN' if x is np.nan else x for x in ctypes]

HOSPITALS, beds, states, htypes, ctypes = (list(t) for t in zip(*sorted(zip(HOSPITALS, beds, states, htypes, ctypes))))
HOSPITALS_SET = sorted(list(set(HOSPITALS)))


with open('dataframe_data/report_categories.csv', newline='') as csvfile:
    categories = csv.reader(csvfile, delimiter=',')
    for row in categories:
        report_categories = row
        report_categories = report_categories[1:]
report_categories.sort()


with open('dataframe_data/sub_categories.csv', newline='') as csvfile:
    categories = csv.reader(csvfile, delimiter=',')
    for row in categories:
        sub_categories = row
sub_categories.sort()

url = 'https://raw.githubusercontent.com/klocey/HCRIS-databuilder/master/provider_data/052043.csv'

main_df = pd.read_csv(url, index_col=[0], header=[0,1,2,3])
main_df = pd.DataFrame(columns = main_df.columns)

print(main_df.shape[1], 'HCRIS features')
print(CMS_NUMS, 'CMS numbers')
print(len(list(set(HOSPITALS))), 'hospitals')

random.seed(42)

COLORS = []
for h in HOSPITALS:
    if 'RUSH UNIVERSITY' in h:
        clr = '#167e04'
    else:
        clr = '#' + "%06x" % random.randint(0, 0xFFFFFF)
    COLORS.append(clr)
    

print(len(sub_categories), 'choosable features')

################# DASH APP CONTROL FUNCTIONS #################################

def myround(n):
    if n == 0:
        return 0
    sgn = -1 if n < 0 else 1
    scale = int(-math.floor(math.log10(abs(n))))
    if scale <= 0:
        scale = 2
    factor = 10**scale
    return sgn*math.floor(abs(n)*factor)/factor


def obs_pred_rsquare(obs, pred):
    return 1 - sum((obs - pred) ** 2) / sum((obs - np.mean(obs)) ** 2)


def description_card1():
    """
    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card1",
        children=[
            html.H5("Hospital cost reports", style={
            'textAlign': 'left',
        }),
           dcc.Markdown("Gaining insights into the Healthcare Cost Report Information System (HCRIS) " +
                        "usually means tackling large, complicated files with expensive software or paying someone " +
                        "else to do it."),
           dcc.Markdown("This app allows you to analyze and download 3,800+ cost related " +
                        "variables for 6,000+ hospitals, for each year since 2010. Get the source code " +
                        "for this app [here] (https://github.com/Rush-Quality-Analytics/hcris-app) and the cost reports " +
                        "for all hospitals [here] (https://github.com/Rush-Quality-Analytics/HCRIS-databuilder/tree/master/provider_data)."),
        ],
    )

def description_card2():
    """
    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card2",
        children=[
            html.H5("Healthcare cost reports", style={
            'textAlign': 'left',
        }),
           html.P("Use this tab to analyze cost report data for an individual hospital." +
                  " Most hospitals will not have data for most variables. Use the top left plot" +
                  " to see which variables your chosen hospital has data for." +
                  " Use the top right plot to examine one variable against another; choose from" +
                  " 4 statistical fits.",
                  style={
            'textAlign': 'left',
        }), 
        ],
    )



def generate_control_card1():
    
    """
    :return: A Div containing controls for graphs.
    """
    
    return html.Div(
        id="control-card1",
        children=[
            
            html.Br(),
            html.H5("1. Filter on the options below"),
            
            html.Div(id='Filterbeds1'),
            dcc.RangeSlider(
                id='beds1',
                #count=1,
                min=0,
                max=2800,
                step=50,
                marks={
                        100: '100',
                        500: '500',
                        1000: '1000',
                        1500: '1500',
                        2000: '2000',
                        2500: '2500',
                    },
                value=[1, 2800],
                ),
            
            html.Br(),
            html.P("Select hospital types"),
            dcc.Dropdown(
                id="hospital_type1",
                options=[{"label": i, "value": i} for i in sorted(list(set(htypes)))],
                multi=True,
                value=sorted(list(set(htypes))),
                style={
                    #'width': '320px', 
                    'font-size': "100%",
                    },
                ),
            
            html.Br(),
            html.P("Select hospital control types"),
            dcc.Dropdown(
                id="control_type1",
                options=[{"label": i, "value": i} for i in sorted(list(set(ctypes)))],
                multi=True,
                value=sorted(list(set(ctypes))),
                style={
                    #'width': '320px', 
                    'font-size': "100%",
                    },
                ),
            
            html.Br(),
            html.P("Select a set of states"),
            dcc.Dropdown(
                id="states-select1",
                options=[{"label": i, "value": i} for i in sorted(list(set(states)))],
                multi=True,
                value=sorted(list(set(states))),
                style={
                    #'width': '320px', 
                    'font-size': "100%",
                    }
            ),
            html.Br(),
            
            html.H5("2. Select hospitals",
                   style={'display': 'inline-block', 'width': '58%'},),
            
            html.I(className="fas fa-question-circle fa-lg", id="target1",
                style={'display': 'inline-block', 'width': '10%', 'color':'#99ccff'},
                ),
            dbc.Tooltip("Hospital names can change over time. This app not only returns data for the hospitals you choose but also returns data for any hospitals with matching CMS numbers.", target="target1",
                style = {'font-size': 12},
                ),
            
            dcc.Dropdown(
                id="hospital-select1",
                options=[{"label": i, "value": i} for i in HOSPITALS_SET],
                multi=True,
                value=None,
                optionHeight=50,
                style={
                    #'width': '320px',
                    'font-size': "90%",
                    }
            ),
            html.Br(),
            
            html.H5("3. Load cost reports",
                   style={'display': 'inline-block', 'width': '64%'},),
            
            html.I(className="fas fa-question-circle fa-lg", id="target2",
                style={'display': 'inline-block', 'width': '10%', 'color':'#99ccff'},
                ),
            dbc.Tooltip("If you add or remove any hospitals, you will need to click the button in order for your changes to take effect.", 
                        target="target2",
                style = {'font-size': 12},
                ),
            
            html.Button('Click to load or update', id='btn1', n_clicks=0,
                style={'width': '80%',
                        'display': 'inline-block',
                        'margin-left': '10%',
                },
                ),
            
            
        ],
    )




def generate_control_card3():
    
    """
    :return: A Div containing controls for graphs.
    """
    
    return html.Div(
        id="control-card3",
        children=[
            
            html.H5("Examine relationships between variables"),
            html.P("Select a category, feature, and scale for your x-variable. "),
            dcc.Dropdown(
                id="categories-select2",
                options=[{"label": i, "value": i} for i in report_categories],
                value=None,
                optionHeight=70,
                style={
                    'width': '250px', 
                    'font-size': "90%",
                    'display': 'inline-block',
                    'border-radius': '15px',
                    #'box-shadow': '1px 1px 1px grey',
                    #'background-color': '#f0f0f0',
                    'padding': '0px',
                    'margin-bottom': '0px',
                    }
            ),
            dcc.Dropdown(
                id="categories-select22",
                options=[{"label": i, "value": i} for i in report_categories],
                value=None,
                optionHeight=70,
                style={
                    'width': '250px', 
                    'font-size': "90%",
                    'display': 'inline-block',
                    'border-radius': '15px',
                    #'box-shadow': '1px 1px 1px grey',
                    #'background-color': '#f0f0f0',
                    'padding': '0px',
                    'margin-left': '10px',
                    }
            ),
            
            dcc.Dropdown(
                    id='x_transform',
                    options=[{"label": i, "value": i} for i in ['linear', 'log10', 'square root']],
                    multi=False, value='linear',
                    style={'width': '120px', 
                            'font-size': "90%",
                            'display': 'inline-block',
                            'border-radius': '15px',
                            #'box-shadow': '1px 1px 1px grey',
                            #'background-color': '#f0f0f0',
                            'padding': '0px',
                            'margin-left': '10px',
                         },
                    ),
            
            
            html.P("Select a category, feature, and scale for your y-variable. "),
            dcc.Dropdown(
                id="categories-select2-2",
                options=[{"label": i, "value": i} for i in report_categories],
                value=None,
                optionHeight=70,
                style={
                    'width': '250px', 
                    'font-size': "90%",
                    'display': 'inline-block',
                    'border-radius': '15px',
                    #'box-shadow': '1px 1px 1px grey',
                    #'background-color': '#f0f0f0',
                    'padding': '0px',
                    'margin-bottom': '0px',
                    }
            ),
            dcc.Dropdown(
                id="categories-select22-2",
                options=[{"label": i, "value": i} for i in report_categories],
                value=None,
                optionHeight=70,
                style={
                    'width': '250px', 
                    'font-size': "90%",
                    'display': 'inline-block',
                    'border-radius': '15px',
                    #'box-shadow': '1px 1px 1px grey',
                    #'background-color': '#f0f0f0',
                    'padding': '0px',
                    'margin-left': '10px',
                    }
            ),
            
            dcc.Dropdown(
                    id='y_transform',
                    options=[{"label": i, "value": i} for i in ['linear', 'log10', 'square root']],
                    multi=False, value='linear',
                    style={'width': '120px', 
                            'font-size': "90%",
                            'display': 'inline-block',
                            'border-radius': '15px',
                            #'box-shadow': '1px 1px 1px grey',
                            #'background-color': '#f0f0f0',
                            'padding': '0px',
                            'margin-left': '10px',
                         },
                    ),
            
            
        ],
        style={
            #'width': '2000px', 
            'font-size': "90%",
            'display': 'inline-block',
            },
    )




def generate_control_card4():
    
    """
    :return: A Div containing controls for graphs.
    """
    
    return html.Div(
        id="control-card4",
        children=[
            
            html.P("Select a model for fitting a trendline"),
            dcc.Dropdown(
                id='trendline-1',
                value='linear',
                style={
                    'width': '200px', 
                    'font-size': "100%",
                    'display': 'inline-block',
                    'border-radius': '15px',
                    #'box-shadow': '1px 1px 1px grey',
                    #'background-color': '#f0f0f0',
                    'padding': '0px',
                    'margin-left': '0px',
                    }
            ),
            
        ],
        style={
            #'width': '2000px', 
            'font-size': "90%",
            'display': 'inline-block',
            },
    )







def generate_control_card5():
    
    """
    :return: A Div containing controls for graphs.
    """
    
    return html.Div(
        id="control-card5",
        children=[
            
            html.H5("Build rate variables and examine them over time"),
            html.P("Select a category and feature for your numerator."),
            dcc.Dropdown(
                id="categories-select3",
                options=[{"label": i, "value": i} for i in report_categories],
                value=None,
                optionHeight=70,
                style={
                    'width': '250px', 
                    'font-size': "90%",
                    'display': 'inline-block',
                    'border-radius': '15px',
                    #'box-shadow': '1px 1px 1px grey',
                    #'background-color': '#f0f0f0',
                    'padding': '0px',
                    'margin-bottom': '0px',
                    }
            ),
            dcc.Dropdown(
                id="categories-select33",
                options=[{"label": i, "value": i} for i in report_categories],
                value=None,
                optionHeight=70,
                style={
                    'width': '250px', 
                    'font-size': "90%",
                    'display': 'inline-block',
                    'border-radius': '15px',
                    #'box-shadow': '1px 1px 1px grey',
                    #'background-color': '#f0f0f0',
                    'padding': '0px',
                    'margin-left': '10px',
                    }
            ),
            
            
            html.P("Select a category and feature for your denominator."),
            dcc.Dropdown(
                id="categories-select3-2",
                options=[{"label": i, "value": i} for i in report_categories],
                value=None,
                optionHeight=70,
                style={
                    'width': '250px', 
                    'font-size': "90%",
                    'display': 'inline-block',
                    'border-radius': '15px',
                    #'box-shadow': '1px 1px 1px grey',
                    #'background-color': '#f0f0f0',
                    'padding': '0px',
                    'margin-bottom': '0px',
                    }
            ),
            dcc.Dropdown(
                id="categories-select33-2",
                options=[{"label": i, "value": i} for i in report_categories],
                value=None,
                optionHeight=70,
                style={
                    'width': '250px', 
                    'font-size': "90%",
                    'display': 'inline-block',
                    'border-radius': '15px',
                    #'box-shadow': '1px 1px 1px grey',
                    #'background-color': '#f0f0f0',
                    'padding': '0px',
                    'margin-left': '10px',
                    }
            ),
            
        ],
        style={
            #'width': '2000px', 
            'font-size': "90%",
            'display': 'inline-block',
            },
    )



#########################################################################################
#############################   DASH APP LAYOUT   #######################################
#########################################################################################    


app.layout = html.Div([
    
    dcc.Store(id='df_tab1', storage_type='memory'),
    html.Div(
        id='url_ls',
        style={'display': 'none'}
        ),
    
    # Banner
    html.Div(
            style={'background-color': '#f9f9f9'},
            id="banner1",
            className="banner",
            children=[html.Img(src=app.get_asset_url("RUSH_full_color.jpg"), 
                               style={'textAlign': 'left'}),
                      html.Img(src=app.get_asset_url("plotly_logo.png"), 
                               style={'textAlign': 'right'})],

        ),
        
    # Left column
    html.Div(
            id="left-column1",
            className="three columns",
            children=[description_card1(), generate_control_card1()],
            style={'width': '24%', 'display': 'inline-block',
                                 'border-radius': '15px',
                                 'box-shadow': '1px 1px 1px grey',
                                 'background-color': '#f0f0f0',
                                 'padding': '10px',
                                 'margin-bottom': '10px',
            },
        ),
        
    # Right column
    html.Div(
            id="right-column1",
            className="eight columns",
            children=[
                
                html.Div(
                    id="map1",
                    children=[
                        html.B("Map of selected hospitals"),
                        html.Hr(),
                        dcc.Graph(id="map_plot1"),
                    ],
                    style={'width': '107%', 'display': 'inline-block',
                                 'border-radius': '15px',
                                 'box-shadow': '1px 1px 1px grey',
                                 'background-color': '#f0f0f0',
                                 'padding': '10px',
                                 'margin-bottom': '10px',
                                 #'fontSize':16
                            },
                ),
                
                
                
                html.Div(
                    [html.A('Download cost reports for your selected hospitals',
                            id="data_link", download="Cost_Reports_Full.csv",
                        href="",
                        target="_blank",
                        style={'fontSize':16}
                        ),
                     html.Br(),
                        ],
                    id="des3",
                    className="mini_container",
                    style={
                        'width': '107%',
                        'display': 'inline-block',
                        'border-radius': '10px',
                        'box-shadow': '1px 1px 1px grey',
                        'background-color': '#f0f0f0',
                        'padding': '10px',
                        'margin-right': '10px',
                        'margin-bottom': '10px',
                        'fontSize':16,
                        'textAlign': 'center',
                        },
                    ),
                
                
                html.Div(
                    id="cost_report1",
                    children=[
                        html.H5("Cost Report Across Fiscal Years"),
                        html.P("Select a category and a feature."),
                        dcc.Dropdown(
                            id="categories-select1",
                            options=[{"label": i, "value": i} for i in report_categories],
                            value=None,
                            optionHeight=70,
                            style={
                                'width': '250px', 
                                'font-size': "90%",
                                'display': 'inline-block',
                                'border-radius': '15px',
                                #'box-shadow': '1px 1px 1px grey',
                                #'background-color': '#f0f0f0',
                                'padding': '0px',
                                'margin-bottom': '0px',
                                }
                        ),
                        dcc.Dropdown(
                            id="categories-select11",
                            options=[{"label": i, "value": i} for i in report_categories],
                            value=None,
                            optionHeight=70,
                            style={
                                'width': '250px', 
                                'font-size': "90%",
                                'display': 'inline-block',
                                'border-radius': '15px',
                                #'box-shadow': '1px 1px 1px grey',
                                #'background-color': '#f0f0f0',
                                'padding': '0px',
                                'margin-left': '10px',
                                }
                        ),
                        
                        html.Hr(),
                        dcc.Graph(id="cost_report_plot1"),
                    ],
                    style={'width': '107%', 'display': 'inline-block',
                                 'border-radius': '15px',
                                 'box-shadow': '1px 1px 1px grey',
                                 'background-color': '#f0f0f0',
                                 'padding': '10px',
                                 'margin-bottom': '10px',
                                 #'fontSize':16
                            },
                ),
                html.Br(),
                html.Br(),
                
                
                
                html.Div(
                    id="cost_report2",
                    children=[
                        generate_control_card3(),
                        dcc.Graph(id="cost_report_plot2"),
                        generate_control_card4(),
                        ],
                    style={
                        'width': '107%',
                        'display': 'inline-block',
                        'border-radius': '15px',
                        'box-shadow': '1px 1px 1px grey',
                        'background-color': '#f0f0f0',
                        'padding': '10px',
                        'margin-bottom': '10px',
                        'height': '770px',
                        },
                ),
                html.Br(),
                #html.Br(),
                
                
                html.Div(
                    id="cost_report3",
                    children=[
                        generate_control_card5(),
                        dcc.Graph(id="cost_report_plot3"),
                        ],
                    style={
                        'width': '107%',
                        'display': 'inline-block',
                        'border-radius': '15px',
                        'box-shadow': '1px 1px 1px grey',
                        'background-color': '#f0f0f0',
                        'padding': '10px',
                        'margin-bottom': '10px',
                        'height': '700px',
                        },
                ),
                html.Br(),
                
            ],
        ),
    ],
)



    
##############################   Callbacks   ############################################
#########################################################################################


@app.callback( # Updated number of beds text
    Output('Filterbeds1', 'children'),
    [
     Input('beds1', 'value'),
     ],
    )
def update_output1(value):
    return 'Number of beds: {}'.format(value)


@app.callback( # Update available sub_categories
    Output('categories-select11', 'options'),
    [
     Input('categories-select1', 'value'),
     Input('df_tab1', "data"),
     ],
    )
def update_output3(value, df):
    
    df2 = main_df.iloc[:, (main_df.columns.get_level_values(2)==value)]
    sub_cat = df2.columns.get_level_values(3).tolist()
    del df2
    
    if df is not None:
        start = timeit.default_timer()
        
        df = pd.read_json(df)
        df.dropna(axis=1, how='all', inplace=True)
        cols1 = list(df)
        cols2 = []
        for c in cols1:
            s = c.split("', ")[-1]
            s = s[1:-2]
            cols2.append(s)
            
        sub_categories = []
        for c in sub_cat:
            if c in cols2:
                sub_categories.append(c)
                
        ex_time = timeit.default_timer() - start
        print("update_output3 executed in "+str(ex_time)) # It returns time in seconds
        
    else:
        sub_categories = sub_cat
    
    
    return [{"label": i, "value": i} for i in sub_categories]


@app.callback( # Select sub-category
    Output('categories-select11', 'value'),
    [
     Input('categories-select11', 'options'),
     ],
    )
def update_output4(available_options):
    try:
        return available_options[0]['value']
    except:
        return 'NUMBER OF BEDS'
    
    
@app.callback(
    Output('hospital-select1', 'options'),
    [
     Input('beds1', 'value'),
     Input('states-select1', 'value'),
     Input('hospital_type1', 'value'),
     Input('control_type1', 'value'),
     ],
    )
def update_hospitals(bed_range, states_vals, htype_vals, ctype_vals):
    start = timeit.default_timer()
    
    low, high = bed_range
    hospitals = []
    for i, h in enumerate(HOSPITALS):
        b = beds[i]
        s = states[i]
        ht = htypes[i]
        ct = ctypes[i]
        if b > low and b < high:
            if s in states_vals and ht in htype_vals:
                if ct in ctype_vals:
                    hospitals.append(h)
            
    hospitals = sorted(list(set(hospitals)))
    
    ex_time = timeit.default_timer() - start
    print("update_hospitals executed in "+str(ex_time)) # It returns time in seconds
    
    return [{"label": i, "value": i} for i in hospitals]


@app.callback(
    Output('url_ls', "children"),
    [Input('btn1', 'n_clicks')],
    [State("hospital-select1", "value"),
     State("hospital-select1", "options"),],
    )
def get_urls(btn1, hospitals, hospital_options):
    start = timeit.default_timer()
    
    options = []
    for h in hospital_options:
        h1 = list(h.values())
        options.append(h1[0])
    
    if hospitals is None or hospitals == []:
        ex_time = timeit.default_timer() - start
        print("get_urls executed in "+str(ex_time)) # It returns time in seconds
        return None
    
    if isinstance(hospitals, str) == True:
        hospitals = [hospitals]
    
    hospitals = list(set(hospitals) & set(options))
    
    if hospitals == []:
        ex_time = timeit.default_timer() - start
        print("get_urls executed in "+str(ex_time)) # It returns time in seconds
        return None
    
    url_ls = []
    for i, val in enumerate(hospitals):
        
        prvdr = re.sub('\ |\?|\.|\!|\/|\;|\:', '', val)
        prvdr = prvdr[prvdr.find("(")+1:prvdr.find(")")]
        
        url = 'https://raw.githubusercontent.com/klocey/HCRIS-databuilder/master/provider_data/' + prvdr + '.csv'
        url_ls.append(url)
    
    ex_time = timeit.default_timer() - start
    print("get_urls executed in "+str(ex_time)) # It returns time in seconds
    
    return url_ls
    

@app.callback(
    Output('df_tab1', "data"),
    [
     Input('url_ls', 'children'),
     Input('df_tab1', "data"),
     ],
    )
def update_df1_tab1(urls, df):
    
    start = timeit.default_timer()
    
    if urls is None or urls is []:
        return None
    
    elif df is None:
        
        for i, url in enumerate(urls):
            
            tdf = pd.read_csv(url, header=[0,1,2,3], index_col=[0])
            tdf[('data url', 'data url', 'data url', 'data url')] = [url] * tdf.shape[0]
            
            if i == 0:
                df = tdf.copy(deep=True)
            else:
                df = pd.concat([df, tdf]) 
                
        df.dropna(axis=1, how='all', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        ex_time = timeit.default_timer() - start
        print("update_df1_tab1 executed in "+str(ex_time)) # It returns time in seconds
        return df.to_json()

    else:
        df = pd.read_json(df)
        df = df[df["('data url', 'data url', 'data url', 'data url')"].isin(urls)]
        df_urls = df["('data url', 'data url', 'data url', 'data url')"].unique()
        
        url_ls = list(set(urls)-set(df_urls))
             
        for i, url in enumerate(url_ls):
            print('new:', url)
            
            tdf = pd.read_csv(url, header=[0,1,2,3], index_col=[0])
            tdf = tdf.to_json()
            tdf = pd.read_json(tdf)
            tdf["('data url', 'data url', 'data url', 'data url')"] = [url] * tdf.shape[0]
            df = pd.concat([df, tdf]) 
    
        df.dropna(axis=1, how='all', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        ex_time = timeit.default_timer() - start
        print("update_df1_tab1 executed in "+str(ex_time)) # It returns time in seconds
        return df.to_json()


@app.callback(
    Output("map_plot1", "figure"),
    [
     Input("df_tab1", "data"),
     Input("hospital-select1", "value"),
     ],
    )
def update_map_plot1(df, h):
    
    figure = go.Figure()
    figure.add_trace(go.Scattermapbox(
    ),
    )
    
    figure.update_layout(
        autosize=True,
        hovermode='closest',
        mapbox=dict(
            accesstoken='pk.eyJ1Ijoia2xvY2V5IiwiYSI6ImNrYm9uaWhoYjI0ZDcycW56ZWExODRmYzcifQ.Mb27BYst186G4r5fjju6Pw',
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=39,
                lon=-98
        ),
        pitch=20,
        zoom=3,
        style='light',
        )
    )

    figure.update_layout(
        height=300, 
        margin={"r":0,"t":0,"l":0,"b":0},
        )

    if df is None:
        return figure
    
    df = pd.read_json(df)
    
    features = list(df)
    if "('Lon', 'Lon', 'Lon', 'Lon')" not in features or "('Lat', 'Lat', 'Lat', 'Lat')" not in features:
        return figure
    
    figure = go.Figure()
    figure.add_trace(go.Scattermapbox(
        lon = df["('Lon', 'Lon', 'Lon', 'Lon')"],
        lat = df["('Lat', 'Lat', 'Lat', 'Lat')"],
        text = df["('Curated Name and Num', 'Curated Name and Num', 'Curated Name and Num', 'Curated Name and Num')"],
               
        marker = dict(
            size = 10,
            color = 'rgb(0, 170, 255)',
            opacity = 0.8,

        ),
        ),
        )
        
    figure.update_layout(
        autosize=True,
        hovermode='closest',
        mapbox=dict(
            accesstoken='pk.eyJ1Ijoia2xvY2V5IiwiYSI6ImNrYm9uaWhoYjI0ZDcycW56ZWExODRmYzcifQ.Mb27BYst186G4r5fjju6Pw',
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=39,
                lon=-98
        ),
        pitch=20,
        zoom=3,
        style='light',
        )
    )

    figure.update_layout(
        height=300, 
        margin={"r":0,"t":0,"l":0,"b":0},
        )
    
    return figure


@app.callback(
    Output("data_link", "href"),
    [
     Input('df_tab1', "data"),
     ],
    )
def update_download(df):
    
    start = timeit.default_timer()
    
    if df is None:
        csv_string = main_df.to_csv(index=False, encoding='utf-8')
        csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_string)
        
        ex_time = timeit.default_timer() - start
        print("update_download executed in "+str(ex_time)) # It returns time in seconds
        return csv_string
            
    df = pd.read_json(df)
    if df.shape[0] == 0:
        csv_string = main_df.to_csv(index=False, encoding='utf-8')
        csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_string)
        ex_time = timeit.default_timer() - start
        print("update_download executed in "+str(ex_time)) # It returns time in seconds
        return csv_string
    
    
    else:
        tdf = main_df.copy(deep=True)
        cols = list(df)
        
        for i, c in enumerate(cols):
            vals = df[c].tolist()
            
            c = list(eval(c))
            tdf[(c[0], c[1], c[2], c[3])] = vals
            
        csv_string = tdf.to_csv(index=True, encoding='utf-8')
        csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_string)
            
    ex_time = timeit.default_timer() - start
    print("update_download executed in "+str(ex_time)) # It returns time in seconds
    return csv_string


@app.callback( # Update Line plot
    Output("cost_report_plot1", "figure"),
    [
     Input('df_tab1', "data"),
     Input('categories-select1', 'value'),
     Input('categories-select11', 'value'),
     ],
    )
def update_cost_report_plot1(df, var1, var2):
    
    start = timeit.default_timer()
    
    if df is None or var1 is None or var1 is None:
        fig = go.Figure(data=go.Scatter(x = [0], y = [0]))

        fig.update_yaxes(title_font=dict(size=14, 
                                         #family='sans-serif', 
                                         color="rgb(38, 38, 38)"))
        fig.update_xaxes(title_font=dict(size=14, 
                                         #family='sans-serif', 
                                         color="rgb(38, 38, 38)"))

        fig.update_layout(title_font=dict(size=14, 
                          color="rgb(38, 38, 38)", 
                          ),
                          showlegend=True,
                          margin=dict(l=100, r=10, b=10, t=10),
                          paper_bgcolor="#f0f0f0",
                          plot_bgcolor="#f0f0f0",
                          )
        
        ex_time = timeit.default_timer() - start
        print("update_cost_report_plot1 executed in "+str(ex_time)) # It returns time in seconds
        
        return fig
         
    df = pd.read_json(df)
    if df.shape[0] == 0:
        fig = go.Figure(data=go.Scatter(x = [0], y = [0]))

        fig.update_yaxes(title_font=dict(size=14, 
                                         #family='sans-serif', 
                                         color="rgb(38, 38, 38)"))
        fig.update_xaxes(title_font=dict(size=14, 
                                         #family='sans-serif', 
                                         color="rgb(38, 38, 38)"))

        fig.update_layout(title_font=dict(size=14, 
                          color="rgb(38, 38, 38)", 
                          ),
                          showlegend=True,
                          margin=dict(l=100, r=10, b=10, t=10),
                          paper_bgcolor="#f0f0f0",
                          plot_bgcolor="#f0f0f0",
                          )
        
        ex_time = timeit.default_timer() - start
        print("update_cost_report_plot1 executed in "+str(ex_time)) # It returns time in seconds
        
        return fig
        
    fig_data = []
    #x = "('PRVDR_NUM', 'Hospital Provider Number', 'HOSPITAL IDENTIFICATION INFORMATION', 'Hospital Provider Number (PRVDR_NUM)')"
    x = "('Curated Name and Num', 'Curated Name and Num', 'Curated Name and Num', 'Curated Name and Num')"
    hospitals = sorted(df[x].unique())
    
    for i, hospital in enumerate(hospitals):
        
        sub_df = df[df[x] == hospital]
        
        #sub_df.sort_values(by=["('FY_END_DT', 'Fiscal Year End Date ', 'HOSPITAL IDENTIFICATION INFORMATION', 'Fiscal Year End Date  (FY_END_DT)')"],
        #                        ascending=[True], inplace=True)
           
        dates = sub_df["('FY_END_DT', 'Fiscal Year End Date', 'HOSPITAL IDENTIFICATION INFORMATION', 'Fiscal Year End Date (FY_END_DT)')"]
            
        str_ = var1 + "', '" + var2 + "')"
        column = [col for col in sub_df.columns if col.endswith(str_)]  
        
        if len(column) == 0:
            fig = go.Figure(data=go.Scatter(x = [0], y = [0]))

            fig.update_yaxes(title_font=dict(size=14, 
                                             #family='sans-serif', 
                                             color="rgb(38, 38, 38)"))
            fig.update_xaxes(title_font=dict(size=14, 
                                             #family='sans-serif', 
                                             color="rgb(38, 38, 38)"))

            fig.update_layout(title_font=dict(size=14, 
                              color="rgb(38, 38, 38)", 
                              ),
                              showlegend=True,
                              margin=dict(l=100, r=10, b=10, t=10),
                              paper_bgcolor="#f0f0f0",
                              plot_bgcolor="#f0f0f0",
                              )
            
            ex_time = timeit.default_timer() - start
            print("update_cost_report_plot1 executed in "+str(ex_time)) # It returns time in seconds
            
            return fig
        
        column = column[0]
        
        obs_y = sub_df[column].tolist()     
        
        #dates, obs_y = map(list, zip(*sorted(zip(dates, obs_y), reverse=False)))
        hospital = str(hospital)
        
        hi = HOSPITALS.index(hospital)
        clr = COLORS[hi]
        
        if len(hospital) > 30:
            hospital = hospital[0:20] + ' ... ' + hospital[-8:]
        
        fig_data.append(
                    go.Scatter(
                        x=dates,
                        y=obs_y,
                        name=hospital,
                        mode='lines+markers',
                        marker=dict(color=clr),
                        #text= text,
                    )
                )
        
        txt_ = '<b>' + var1 + '<b>'
        var2b = re.sub("\(.*?\)|\[.*?\]","", var2)
        
        if len(var2b) > 40:
            var_ls = []
            for j in range(0, len(var2b), 40):
                var_ls.append(var2b[j : j + 40])
            
            
            for j in var_ls:
                txt_ = txt_ + '<br>' + j 
        else:
            txt_ = txt_ + '<br>' + var2b 
            
        figure = go.Figure(
            data=fig_data,
            layout=go.Layout(
                transition = {'duration': 500},
                xaxis=dict(
                    title=dict(
                        text="<b>Date</b>",
                        font=dict(
                            family='"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                            " Helvetica, Arial, sans-serif",
                            size=18,
                        ),
                    ),
                    rangemode="tozero",
                    zeroline=True,
                    showticklabels=True,
                ),
                
                yaxis=dict(
                    title=dict(
                        text=txt_,
                        font=dict(
                            family='"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                            " Helvetica, Arial, sans-serif",
                            size=14,
                            
                        ),
                    ),
                    rangemode="tozero",
                    zeroline=True,
                    showticklabels=True,
                    
                ),
                
                margin=dict(l=100, r=30, b=10, t=40),
                showlegend=True,
                height=400,
                paper_bgcolor="#f0f0f0",
                plot_bgcolor="#f0f0f0",
            ),
        )
        
        figure.update_layout(
            legend=dict(
                traceorder="normal",
                font=dict(
                    size=10,
                    color="rgb(38, 38, 38)"
                ),
                
            )
        )    
    
    del df
    del hospitals
    del fig_data
    del dates
    del x
    
    ex_time = timeit.default_timer() - start
    print("update_cost_report_plot1 executed in "+str(ex_time)) # It returns time in seconds
    
    return figure
    

@app.callback( # Update available sub_categories
    Output('categories-select22', 'options'),
    [
     Input('categories-select2', 'value'),
     Input('df_tab1', "data"),
     ],
    )
def update_output7(value, df):
    
    start = timeit.default_timer()
    
    df2 = main_df.iloc[:, (main_df.columns.get_level_values(2)==value)]
    sub_cat = df2.columns.get_level_values(3).tolist()
    del df2
    
    if df is not None:
        df = pd.read_json(df)
        df.dropna(axis=1, how='all', inplace=True)
        cols1 = list(df)
        cols2 = []
        for c in cols1:
            s = c.split("', ")[-1]
            s = s[1:-2]
            cols2.append(s)
            
        sub_categories = []
        for c in sub_cat:
            if c in cols2:
                sub_categories.append(c)
    else:
        sub_categories = sub_cat
    
    ex_time = timeit.default_timer() - start
    print("update_output7 executed in "+str(ex_time)) # It returns time in seconds
    
    return [{"label": i, "value": i} for i in sub_categories]


@app.callback( # Select sub-category
    Output('categories-select22', 'value'),
    [
     Input('categories-select22', 'options'),
     ],
    )
def update_output8(available_options):
    try:
        return available_options[0]['value']
    except:
        return 'NUMBER OF BEDS'
    
    
@app.callback( # Update available sub_categories
    Output('categories-select22-2', 'options'),
    [
     Input('categories-select2-2', 'value'),
     Input('df_tab1', "data"),
     ],
    )
def update_output9(value, df):

    start = timeit.default_timer()

    df2 = main_df.iloc[:, (main_df.columns.get_level_values(2)==value)]
    sub_cat = df2.columns.get_level_values(3).tolist()
    del df2
    
    if df is not None:
        df = pd.read_json(df)
        df.dropna(how='all', axis=1, inplace=True)
        
        cols1 = list(df)
        cols2 = []
        for c in cols1:
            s = c.split("', ")[-1]
            s = s[1:-2]
            cols2.append(s)
            
        sub_categories = []
        for c in sub_cat:
            if c in cols2:
                sub_categories.append(c)
    else:
        sub_categories = sub_cat
    
    #sub_categories = sorted(sub_categories)
    ex_time = timeit.default_timer() - start
    print("update_output9 executed in "+str(ex_time)) # It returns time in seconds

    return [{"label": i, "value": i} for i in sub_categories]


@app.callback( # Select sub-category
    Output('categories-select22-2', 'value'),
    [
     Input('categories-select22-2', 'options'),
     ],
    )
def update_output10(available_options):
    try:
        return available_options[0]['value']
    except:
        return 'NUMBER OF BEDS'
    

@app.callback( # Update available sub_categories
    Output('categories-select33', 'options'),
    [
     Input('categories-select3', 'value'),
     Input('df_tab1', "data"),
     ],
    )
def update_output11(value, df):
    
    start = timeit.default_timer()
    
    df2 = main_df.iloc[:, (main_df.columns.get_level_values(2)==value)]
    sub_cat = df2.columns.get_level_values(3).tolist()
    del df2
    
    if df is not None:
        df = pd.read_json(df)
        df.dropna(axis=1, how='all', inplace=True)
        cols1 = list(df)
        cols2 = []
        for c in cols1:
            s = c.split("', ")[-1]
            s = s[1:-2]
            cols2.append(s)
            
        sub_categories = []
        for c in sub_cat:
            if c in cols2:
                sub_categories.append(c)
    else:
        sub_categories = sub_cat
    
    ex_time = timeit.default_timer() - start
    print("update_output11 executed in "+str(ex_time)) # It returns time in seconds

    return [{"label": i, "value": i} for i in sub_categories]


@app.callback( # Select sub-category
    Output('categories-select33', 'value'),
    [
     Input('categories-select33', 'options'),
     ],
    )
def update_output12(available_options):
    try:
        return available_options[0]['value']
    except:
        return 'NUMBER OF BEDS'    
    

@app.callback( # Update available sub_categories
    Output('categories-select33-2', 'options'),
    [
     Input('categories-select3-2', 'value'),
     Input('df_tab1', "data"),
     ],
    )
def update_output13(value, df):
    
    start = timeit.default_timer()
    
    df2 = main_df.iloc[:, (main_df.columns.get_level_values(2)==value)]
    sub_cat = df2.columns.get_level_values(3).tolist()
    del df2
    
    if df is not None:
        df = pd.read_json(df)
        df.dropna(how='all', axis=1, inplace=True)
        
        cols1 = list(df)
        cols2 = []
        for c in cols1:
            s = c.split("', ")[-1]
            s = s[1:-2]
            cols2.append(s)
            
        sub_categories = []
        for c in sub_cat:
            if c in cols2:
                sub_categories.append(c)
    else:
        sub_categories = sub_cat
    
    ex_time = timeit.default_timer() - start
    print("update_output13 executed in "+str(ex_time)) # It returns time in seconds

    #sub_categories = sorted(sub_categories)
    return [{"label": i, "value": i} for i in sub_categories]


@app.callback( # Select sub-category
    Output('categories-select33-2', 'value'),
    [
     Input('categories-select33-2', 'options'),
     ],
    )
def update_output14(available_options):
    try:
        return available_options[0]['value']
    except:
        return 'NUMBER OF BEDS'
  

@app.callback( # Update Line plot
    Output("cost_report_plot2", "figure"),
    [
     Input('categories-select2', 'value'),
     Input('categories-select22', 'value'),
     Input('categories-select2-2', 'value'),
     Input('categories-select22-2', 'value'),
     Input('x_transform', 'value'),
     Input('y_transform', 'value'),
     Input('trendline-1', 'value'),
     ],
    [State("df_tab1", "data")],
    )
def update_cost_report_plot2(xvar1, xvar2, yvar1, yvar2, xscale, yscale, model, df):
    start = timeit.default_timer()
    
    if df is None or xvar1 is None or xvar2 is None or yvar1 is None or yvar2 is None or yvar2 == 'NUMBER OF BEDS':
            
        tdf = pd.DataFrame(columns=['x', 'y'])
        tdf['x'] = [0]
        tdf['y'] = [0]
        fig = px.scatter(tdf, x = 'x', y = 'y')

        fig.update_yaxes(title_font=dict(size=14, 
                                     #family='sans-serif', 
                                     color="rgb(38, 38, 38)"))
        fig.update_xaxes(title_font=dict(size=14, 
                                     #family='sans-serif', 
                                     color="rgb(38, 38, 38)"))
        fig.update_layout(title_font=dict(size=14, 
                      color="rgb(38, 38, 38)", 
                      ),
                      showlegend=True,
                      margin=dict(l=100, r=10, b=10, t=10),
                      paper_bgcolor="#f0f0f0",
                      plot_bgcolor="#f0f0f0",
                      )
        
        ex_time = timeit.default_timer() - start
        print("update_cost_report_plot2 executed in "+str(ex_time)) # It returns time in seconds
        return fig
            
    
    df = pd.read_json(df)
    
    fig_data = []
    
    str_1 = xvar1 + "', '" + xvar2 + "')"
    str_2 = yvar1 + "', '" + yvar2 + "')"
    
    column1 = [col for col in df.columns if col.endswith(str_1)]
    column2 = [col for col in df.columns if col.endswith(str_2)]
    
    if len(column1) == 0 or len(column2) == 0:
        
        tdf = pd.DataFrame(columns=['x', 'y'])
        tdf['x'] = [0]
        tdf['y'] = [0]
        fig = px.scatter(tdf, x = 'x', y = 'y')

        fig.update_yaxes(title_font=dict(size=14, 
                                     #family='sans-serif', 
                                     color="rgb(38, 38, 38)"))
        fig.update_xaxes(title_font=dict(size=14, 
                                     #family='sans-serif', 
                                     color="rgb(38, 38, 38)"))
        fig.update_layout(title_font=dict(size=14, 
                      color="rgb(38, 38, 38)", 
                      ),
                      showlegend=True,
                      margin=dict(l=100, r=10, b=10, t=10),
                      paper_bgcolor="#f0f0f0",
                      plot_bgcolor="#f0f0f0",
                      )
        
        ex_time = timeit.default_timer() - start
        print("update_cost_report_plot2 executed in "+str(ex_time)) # It returns time in seconds
        return fig
    
    
    hospitals = sorted(df["('Curated Name and Num', 'Curated Name and Num', 'Curated Name and Num', 'Curated Name and Num')"].unique())                 
    
    fig_data = []
    for i, hospital in enumerate(hospitals):
        
        tdf = df[df["('Curated Name and Num', 'Curated Name and Num', 'Curated Name and Num', 'Curated Name and Num')"] == hospital]
        
        column1 = [col for col in tdf.columns if col.endswith(str_1)]
        column2 = [col for col in tdf.columns if col.endswith(str_2)]
        
        column1 = column1[0]
        x = tdf[column1].tolist()
        
        column2 = column2[0]
        y = tdf[column2].tolist()
        
        dates = tdf["('FY_END_DT', 'Fiscal Year End Date', 'HOSPITAL IDENTIFICATION INFORMATION', 'Fiscal Year End Date (FY_END_DT)')"]
        
        names = tdf["('Curated Name and Num', 'Curated Name and Num', 'Curated Name and Num', 'Curated Name and Num')"]
        
        text = names + '<br>' + dates.astype(str)
        
        x, y, dates, names = map(list, zip(*sorted(zip(x, y, dates, names), reverse=False)))
        
        if xscale == 'log10' and np.nanmin(x) >= 0:
            x = np.log10(x)
        elif xscale == 'square root' and np.nanmin(x) >= 0:
            x = np.sqrt(x)
            
        if yscale == 'log10' and np.nanmin(y) >= 0:
            y = np.log10(y)
        elif yscale == 'square root' and np.nanmin(y) >= 0:
            y = np.sqrt(y)
            
        hi = HOSPITALS.index(hospital)
        clr = COLORS[hi]
        
        if len(hospital) > 30:
            hospital = hospital[0:20] + ' ... ' + hospital[-8:]
        
        fig_data.append(
                    go.Scatter(
                        x=x,
                        y=y,
                        name=hospital,
                        mode='markers',
                        marker=dict(color=clr),
                        text= text,
                    )
                )
    
    column1 = [col for col in df.columns if col.endswith(str_1)]
    column2 = [col for col in df.columns if col.endswith(str_2)]
    
    dates = df["('FY_END_DT', 'Fiscal Year End Date', 'HOSPITAL IDENTIFICATION INFORMATION', 'Fiscal Year End Date (FY_END_DT)')"].tolist()
    
    column1 = column1[0]
    x = df[column1].tolist()
    
    column2 = column2[0]
    y = df[column2].tolist()
    
    if xscale == 'log10' and np.nanmin(x) >= 0:
        x = np.log10(x)
    elif xscale == 'square root' and np.nanmin(x) >= 0:
        x = np.sqrt(x)
        
    if yscale == 'log10' and np.nanmin(y) >= 0:
        y = np.log10(y)
    elif yscale == 'square root' and np.nanmin(y) >= 0:
        y = np.sqrt(y)
        
    tdf = pd.DataFrame(columns = ['x', 'y'])
    tdf['x'] = x
    tdf['y'] = y
    tdf['dates'] = dates
    tdf.dropna(how='any', inplace=True)
    tdf.sort_values(by='x', inplace=True, ascending=True)
    
    x = tdf['x']
    y = tdf['y']
    dates = tdf['dates'].tolist()
        
    ty = []
    r2 = ''
        
    if x.tolist() == [] or y.tolist() == []:
        slope, intercept, r_value, p_value, std_err = np.nan, np.nan, np.nan, np.nan, np.nan
        r2 = np.nan
    
        
    tdf.replace([np.inf, -np.inf], np.nan, inplace=True)
    tdf.dropna(how='any', inplace=True)
        
    x_o = tdf['x'].values.tolist()
    y_o = tdf['y'].values.tolist()
    x_o, y_o = zip(*sorted(zip(x_o, y_o)))
    
    x_o = np.array(x_o)
    y_o = np.array(y_o)
    
    #Create single dimension
    x = x_o[:, np.newaxis]
    y = y_o[:, np.newaxis]

    inds = x.ravel().argsort()  # Sort x values and get index
    x = x.ravel()[inds].reshape(-1, 1)
    y = y[inds] #Sort y according to x sorted index
    
    d = int()
    if model == 'linear': d = 1
    elif model == 'quadratic': d = 2
    elif model == 'cubic': d = 3
    
    polynomial_features = PolynomialFeatures(degree = d)
    xp = polynomial_features.fit_transform(x)
        
    model = sm.OLS(y, xp).fit()
    ypred = model.predict(xp)
    ypred = ypred.tolist()
    
    poly_coefs = model.params[1:].tolist()
    poly_coefs.reverse()
    
    poly_exponents = list(range(1, len(poly_coefs)+1))
    poly_exponents.reverse()
    
    eqn = 'y = '
    for i, p in enumerate(poly_coefs):
        exp = poly_exponents[i]
        
        if exp == 1:
            exp = 'x'
        elif exp == 2:
            exp = 'x²'
        elif exp == 3:
            exp = 'x³'
        
        if i == 0:
            p = myround(p)
            eqn = eqn + str(p) + exp
            
        else:
            if p >= 0:
                p = myround(p)
                eqn = eqn + ' + ' + str(p) + exp
            else:
                p = myround(p)
                eqn = eqn + ' - ' + str(np.abs(p)) + exp
    
    b = model.params[0]
    if b >= 0:
        b = myround(b)
        eqn = eqn + ' + ' + str(b)
    else:
        b = myround(b)
        eqn = eqn + ' - ' + str(np.abs(b))
        
    r2 = model.rsquared_adj
    r2_adj = model.rsquared_adj
    aic = model.aic
    bic = model.bic
    fp = model.f_pvalue
    llf = model.llf
    
    st, data, ss2 = summary_table(model, alpha=0.05)
    fittedvalues = data[:, 2]
    predict_mean_se  = data[:, 3]
    predict_mean_ci_low, predict_mean_ci_upp = data[:, 4:6].T
    predict_ci_low, predict_ci_upp = data[:, 6:8].T
    
    outlier_y = []
    outlier_x = []
    nonoutlier_y = []
    nonoutlier_x = []
    for i, yi in enumerate(y_o):
        if yi > predict_ci_upp[i] or yi < predict_ci_low[i]:
            outlier_y.append(yi)
            outlier_x.append(x_o[i])
        else:
            nonoutlier_y.append(yi)
            nonoutlier_x.append(x_o[i])
            
    clr = "#3399ff"
    #x, y, ypred = zip(*sorted(zip(x, y, ypred)))
    
    fig_data.append(go.Scatter(
                        x = nonoutlier_x,
                        y = nonoutlier_y,
                        name = 'Non-outliers',
                        mode = "markers",
                        opacity = 1.0,
                        marker = dict(size=10,
                                    color=clr,
                                    symbol="diamond-open",
                                    )
                    )
                )
                
    fig_data.append(go.Scatter(
            x = outlier_x,
            y = outlier_y,
            name = 'Outliers',
            mode = "markers",
            opacity = 1.0,
            marker = dict(size=10,
                        color="#ff0000",
                        symbol="diamond-open",
                        )
        )
    )
    
    fig_data.append(
                go.Scatter(
                    x = x_o,
                    y = ypred,
                    mode = "lines",
                    name = 'fitted: r2 = <sup>'+str(np.round(r2, 3))+'</sup>',
                    opacity = 0.75,
                    line = dict(color = clr, width = 2),
                )
            )
    
    fig_data.append(
        go.Scatter(
            x = x_o,
            y = predict_mean_ci_upp,
            mode = "lines",
            name = 'upper 95 CI',
            opacity = 0.75,
            line = dict(color = clr, width = 2, dash='dash'),
        )
    )
    
    fig_data.append(
        go.Scatter(
            x = x_o,
            y = predict_mean_ci_low,
            mode = "lines",
            name = 'lower 95 CI',
            opacity = 0.75,
            line = dict(color = clr, width = 2, dash='dash'),
        )
    )
    
    fig_data.append(
        go.Scatter(
            x = x_o,
            y = predict_ci_upp,
            mode = "lines",
            name = 'upper 95 PI',
            opacity = 0.75,
            line = dict(color = clr, width = 2, dash='dot'),
        )
    )
    
    fig_data.append(
        go.Scatter(
            x = x_o,
            y = predict_ci_low,
            mode = "lines",
            name = 'lower 95 PI',
            opacity = 0.75,
            line = dict(color = clr, width = 2, dash='dot'),
        )
    )
    
    txt1 = '<b>' + xvar1 + '<b>'
    xvar2b = re.sub("\(.*?\)|\[.*?\]","", xvar2)
    if len(xvar2b) > 40:
        var_ls = []
        for j in range(0, len(xvar2b), 40):
            var_ls.append(xvar2b[j : j + 40])
        
        for j in var_ls:
            txt1 = txt1 + '<br>' + j 
    else:
        txt1 = txt1 + '<br>' + xvar2b
    
        
    txt2 = '<b>' + yvar1 + '<b>'
    yvar2b = re.sub("\(.*?\)|\[.*?\]","", yvar2)
    if len(yvar2b) > 40:
        var_ls = []
        for j in range(0, len(yvar2b), 40):
            var_ls.append(yvar2b[j : j + 40])
        
        for j in var_ls:
            txt2 = txt2 + '<br>' + j 
    else:
        txt2 = txt2 + '<br>' + yvar2b
        
    figure = go.Figure(
        data=fig_data,
        layout=go.Layout(
            transition = {'duration': 500},
            xaxis=dict(
                title=dict(
                    text=txt1,
                    font=dict(
                        family='"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                        " Helvetica, Arial, sans-serif",
                        size=14,
                    ),
                ),
                #rangemode="tozero",
                zeroline=True,
                showticklabels=True,
            ),
                
            yaxis=dict(
                title=dict(
                    text=txt2,
                    font=dict(
                        family='"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                        " Helvetica, Arial, sans-serif",
                        size=14,
                            
                    ),
                ),
                #rangemode="tozero",
                zeroline=True,
                showticklabels=True,
                    
            ),
                
            margin=dict(l=80, r=10, b=80, t=60),
            showlegend=True,
            height=500,
            paper_bgcolor="#f0f0f0",
            plot_bgcolor="#f0f0f0",
        ),
    )
        
    if x.tolist() != [] and y.tolist() != []:
        figure.update_layout(
            title="Percent variation explained by the model: " + str(r2),
            font=dict(
                size=10,
                color="rgb(38, 38, 38)"
                ),
            )
    
    figure.update_layout(
        legend=dict(
            traceorder="normal",
            font=dict(
                size=10,
                color="rgb(38, 38, 38)"
            ),
            
        )
    )    
    
    del tdf
    del df
    del hospitals
    del fig_data
    del dates
    del names
    del y
    del x
    del txt1
    del txt2
    
    ex_time = timeit.default_timer() - start
    print("update_cost_report_plot2 executed in "+str(ex_time)) # It returns time in seconds

    return figure


@app.callback( # Update Line plot
    Output("cost_report_plot3", "figure"),
    [
     Input("df_tab1", "data"),
     Input('categories-select3', 'value'),
     Input('categories-select33', 'value'),
     Input('categories-select3-2', 'value'),
     Input('categories-select33-2', 'value'),
     ],
    )
def update_cost_report_plot3(df, numer1, numer2, denom1, denom2):
    
    start = timeit.default_timer()
    
    if df is None or numer1 is None or numer2 is None or denom1 is None or denom2 is None or denom2 == 'NUMBER OF BEDS':
            
        tdf = pd.DataFrame(columns=['x', 'y'])
        tdf['x'] = [0]
        tdf['y'] = [0]
        fig = px.scatter(tdf, x = 'x', y = 'y')

        fig.update_yaxes(title_font=dict(size=14, 
                                     #family='sans-serif', 
                                     color="rgb(38, 38, 38)"))
        fig.update_xaxes(title_font=dict(size=14, 
                                     #family='sans-serif', 
                                     color="rgb(38, 38, 38)"))
        fig.update_layout(title_font=dict(size=14, 
                      color="rgb(38, 38, 38)", 
                      ),
                      showlegend=True,
                      margin=dict(l=100, r=10, b=10, t=10),
                      paper_bgcolor="#f0f0f0",
                      plot_bgcolor="#f0f0f0",
                      )
        
        ex_time = timeit.default_timer() - start
        print("update_cost_report_plot3 executed in "+str(ex_time)) # It returns time in seconds
        return fig
            
    
    df = pd.read_json(df)
    
    fig_data = []
    
    #df['years'] = pd.to_datetime(dates).dt.year
    #headers = list(set(list(df)))
    
    numer = numer1 + "', '" + numer2 + "')"
    denom = denom1 + "', '" + denom2 + "')"
    
    column1 = [col for col in df.columns if col.endswith(numer)]
    column2 = [col for col in df.columns if col.endswith(denom)]
    
    if len(column1) == 0 or len(column2) == 0:
        
        tdf = pd.DataFrame(columns=['x', 'y'])
        tdf['x'] = [0]
        tdf['y'] = [0]
        fig = px.scatter(tdf, x = 'x', y = 'y')

        fig.update_yaxes(title_font=dict(size=14, 
                                     #family='sans-serif', 
                                     color="rgb(38, 38, 38)"))
        fig.update_xaxes(title_font=dict(size=14, 
                                     #family='sans-serif', 
                                     color="rgb(38, 38, 38)"))
        fig.update_layout(title_font=dict(size=14, 
                      color="rgb(38, 38, 38)", 
                      ),
                      showlegend=True,
                      margin=dict(l=100, r=10, b=10, t=10),
                      paper_bgcolor="#f0f0f0",
                      plot_bgcolor="#f0f0f0",
                      )
        ex_time = timeit.default_timer() - start
        print("update_cost_report_plot3 executed in "+str(ex_time)) # It returns time in seconds
        return fig
    
    
    hospitals = sorted(df["('Curated Name and Num', 'Curated Name and Num', 'Curated Name and Num', 'Curated Name and Num')"].unique())                 
    
    fig_data = []
    for hospital in hospitals:
        
        name_var = "('Curated Name and Num', 'Curated Name and Num', 'Curated Name and Num', 'Curated Name and Num')"
        tdf = df[df[name_var] == hospital]
        
        date_var = "('FY_END_DT', 'Fiscal Year End Date', 'HOSPITAL IDENTIFICATION INFORMATION', 'Fiscal Year End Date (FY_END_DT)')"
        tdf.sort_values(by=date_var, inplace=True, ascending=True)
        
        column1 = [col for col in tdf.columns if col.endswith(numer)]
        column2 = [col for col in tdf.columns if col.endswith(denom)]
        
        column1 = column1[0]
        column2 = column2[0]
        tdf['y'] = tdf[column1]/tdf[column2]
        tdf = tdf.filter(items=['y', date_var, name_var], axis=1)
        tdf.dropna(how='any', inplace=True)
        
        dates = tdf["('FY_END_DT', 'Fiscal Year End Date', 'HOSPITAL IDENTIFICATION INFORMATION', 'Fiscal Year End Date (FY_END_DT)')"]
        names = tdf["('Curated Name and Num', 'Curated Name and Num', 'Curated Name and Num', 'Curated Name and Num')"]
        y = tdf['y']
        #dates, y, x = map(list, zip(*sorted(zip(dates, y, x), reverse=False)))
        #x, y, dates = map(list, zip(*sorted(zip(x, y, dates), reverse=False)))
        
        text = names + '<br>' + dates.astype(str)
        
        hi = HOSPITALS.index(hospital)
        clr = COLORS[hi]
        
        if len(hospital) > 30:
            hospital = hospital[0:20] + ' ... ' + hospital[-8:]
            
        fig_data.append(
                    go.Scatter(
                        x=dates,
                        y=y,
                        name=hospital,
                        mode='lines+markers',
                        marker=dict(color=clr),
                        text= text,
                    )
                )
    
    figure = go.Figure(
        data=fig_data,
        layout=go.Layout(
            transition = {'duration': 500},
            xaxis=dict(
                title=dict(
                    text="<b>Date</b>",
                    font=dict(
                        family='"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                        " Helvetica, Arial, sans-serif",
                        size=18,
                    ),
                ),
                #rangemode="tozero",
                zeroline=True,
                showticklabels=True,
            ),
                
            yaxis=dict(
                title=dict(
                    text="<b>" + numer2 + ' / ' + denom2 + "</b>",
                    font=dict(
                        family='"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                        " Helvetica, Arial, sans-serif",
                        size=14,
                            
                    ),
                ),
                #rangemode="tozero",
                zeroline=True,
                showticklabels=True,
                    
            ),
                
            margin=dict(l=80, r=10, b=80, t=60),
            showlegend=True,
            height=500,
            paper_bgcolor="#f0f0f0",
            plot_bgcolor="#f0f0f0",
        ),
    )
        
    figure.update_layout(
        legend=dict(
            traceorder="normal",
            font=dict(
                size=10,
                color="rgb(38, 38, 38)"
            ),
            
        )
    )    
    
    del tdf
    del df
    del hospitals
    del fig_data
    del dates
    del names
    del y
    del date_var
    del name_var
    
    ex_time = timeit.default_timer() - start
    print("update_cost_report_plot3 executed in "+str(ex_time)) # It returns time in seconds
    return figure


@app.callback( # Update available sub_categories
    Output('trendline-1', 'options'),
    [
     Input('trendline-1', 'value'),
     ],
    )
def update_output15(value):
    options = ['linear', 'quadratic', 'cubic']
    
    return [{"label": i, "value": i} for i in options]



#########################################################################################


# Run the server
if __name__ == "__main__":
    app.run_server(host='0.0.0.0', debug = True) # modified to run on linux server

