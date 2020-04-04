# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

from datetime import datetime as dt
from scipy import stats

########################################################################################
# Sub functions
########################################################################################
def hovertext(time, param, suffix):
    hover_string = '{}<br>{:.3f}{}'.format(time, param, suffix)
    return hover_string

def hex_to_rgb(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = hex_color * 2
    return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)

# Resample the selected parameters
def resample_params(file_path, interval, fields):
    """Resample the input csv file based on the specified time interval,
    and output the selected resampled parameters

    Input arguments:
        file_path: str, input csv file path
        interval: str, time intervals in minutes, e.g., "15Min", "60Min"
        fields: list, selected fields, e.g., ["E", "P", "P1"]

    Output arguments:
        param_resampled: pandas dataframe, resampled parameters 
    """
    df = pd.read_csv(file_path, header=0,infer_datetime_format=True, 
                        parse_dates=['time_utc'], index_col=['time_utc'])
    param_resampled = df[fields].resample(interval, base=0) 

    return param_resampled

# Calculate Student's t test
def calc_ttest(sample_x, sample_y):
    """Calculate Student's t test for two sets of samples.
    sample_x: resampled time series data with time indexed groups
    sample_y: resampled time series data Y with the same time indices for all groups

    returns:
        pvalue_all: p values resulted from the comparision of pairwise groups 
    """
    # find the group names
    group_names = []
    time_idx = []
    for name, group_vals in sample_x:
        x = name.strftime('%H:%M')
        time_idx.append(x)
        group_names.append(name)

    # t_all = []  # t value
    pvalue_all = []
    for i, name in enumerate(group_names):
        sample_x_i = sample_x.get_group(name).values
        sample_y_i = sample_y.get_group(name).values

        # apply two samples t test
        t, pvalue = stats.ttest_ind(sample_x_i, sample_y_i)
        # print("t: {}, pvalue: {}\n".format(t, pvalue))
        
        # t_all.append(t)
        pvalue_all.append(pvalue)

    return pvalue_all

########################################################################################
# Import data
########################################################################################

uuid_all = {}
uuid_all['b-kitchen'] = ['001ec08959bf0004']
uuid_all['i-711-denmark'] = ['0004a3a584220001','0004a3a584220002','0004a3a584220003',
                            '0004a3a584220004','0004a3a584220005']
uuid_all['i-velux'] = ['0004a3a5b5cd0001','0004a3a5b5cd0002','0004a3a5b5cd0003',
                        '0004a3a5b5cd0004','0004a3a5b5cd0005','0004a3a5b5cd0006',
                        '0004a3a5b5cd0007','0004a3a5b5cd0008']
uuid_all['s-kfc-malaysia'] = ['001ec0b998ca0001','001ec0b998ca0002','001ec0b998ca0003',
                            '001ec0b998ca0004']

field_all = {}
field_all['15Min'] = ['P','P1','P2','P3']
field_all['30Min'] = ['P','P1','P2','P3','E','E1','E2','E3']

# data directory
DATASETS_FOLDER = "/media/zeliang/Data/Datasets"  #
DIR_MAPPING = {
    'b-kitchen': 'BEST',
    'i-711-denmark': 'IQEnergy',
    'i-velux': 'IQEnergy',
    's-kfc-malaysia': 'SeidoSolutions'
}

DATA_DIR = os.path.join(DATASETS_FOLDER, 's3_results', 'combined') 
training_data = 's3-stats-results-training.csv'
df = pd.read_csv(DATA_DIR + "/" + training_data)
# time_idx1 = df.columns[6:].to_list()  # time columns start from 6
time_idx1 = df.columns[7:].to_list()  

# convert power from Watts to kilo-Watts
df.loc[:, time_idx1] = df.loc[:, time_idx1].div(1000)

testing_data = 's3-stats-results-testing.csv'
df_testing = pd.read_csv(DATA_DIR + "/" + testing_data)
time_idx2 = df_testing.columns[5:].to_list()  # 
# convert power from Watts to kilo-Watts
df_testing.loc[:, time_idx2] = df_testing.loc[:, time_idx2].div(1000)

########################################################################################
# Prepare the data for plotting
########################################################################################
color_alpha = [
    ('#7f7f7f', 0.5),  # middle gray
    ('#8c599b', 0.5), # purple
    ('#1f77e4', 0.9),  # muted blue
    ('#2cd02c', 0.8),   # cooked asparagus green
    ('#d62758', 0.8) # red
    ]

colors = [f"rgba{(*hex_to_rgb(c[0]), c[1])}" for c in color_alpha]

########################################################################################
# Define dash object
########################################################################################
app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)
app.title = 'PMS 1.0'
server = app.server

# App Layout
logscale_val = [10**(i) for i in range(-12,1,3)]
app.layout = html.Div(
    children=[
        # Error Message
        html.Div(id="error-message"),
        html.Div(id="error-message2"),
        # Top Banner
        html.Div(
            className="top-banner row",
            children=[
                html.Div(
                    className="div-logo-left",
                    children=html.Img(
                        className="logo-left", src=app.get_asset_url("pasty-logo.png")
                    ),
                ),
                html.H2(className="h2-title", children="Exception Reporting Demo"),
                html.Div(
                    className="div-logo",
                    children=html.Img(
                        className="logo", src=app.get_asset_url("best-logo-3.png")
                    ),
                ),
                html.H2(className="h2-title-mobile", children="Exception Reporting Demo"),
            ],
        ),
        # Body of the App
        html.Div(
            className="row app-body",
            children=[
                # User Controls
                html.Div(
                    className="three columns card",
                    children=[
                        html.Div(
                            className="bg-white-control-panel user-control",
                            children=[
                                html.Div(
                                    className="padding-top-bot",
                                    children=[
                                        html.H6("Case study"),
                                        dcc.Dropdown(
                                            id="case-study",
                                            options=[
                                                {'label': 'Velux (Denmark)', 'value': 'i-velux'},
                                                {'label': 'BEST Kitchen', 'value': 'b-kitchen'},
                                                {'label': '7 Eleven (Denmark)', 'value': 'i-711-denmark'},
                                                {'label': 'KFC (Malaysia)', 'value': 's-kfc-malaysia'}
                                            ],
                                            value="b-kitchen",
                                            ),
                                    ],
                                ),

                                html.Div(
                                    className="padding-top-bot",
                                    children=[
                                        html.H6("Eniscope UUID"),
                                        dcc.Dropdown(
                                            id="eniscope-uuid",
                                            ),
                                    ],
                                ),

                                html.Div(
                                    className="padding-top-bot",
                                    children=[
                                        html.H6("Field"),
                                        dcc.Dropdown(
                                            id="field-opt",
                                            ),
                                    ],
                                ),

                                html.Div(
                                    className="padding-top-bot",
                                    children=[
                                        html.H6("Year"),
                                        dcc.Dropdown(
                                            id="year-dropdown",
                                            options=[
                                                {"label": "2016", "value": "2016"},
                                                {"label": "2017", "value": "2017"},
                                                {"label": "2018", "value": "2018"},
                                                {"label": "2019", "value": "2019"},
                                            ],
                                            value="2018",
                                            ),
                                    ],
                                ),

                                html.Div(
                                    className="padding-top-bot",
                                    children=[
                                        html.H6("Day"),
                                        dcc.Dropdown(
                                            id="day-dropdown",
                                            options=[
                                                {"label": "Monday", "value": "mon"},
                                                {"label": "Tuesday", "value": "tue"},
                                                {"label": "Wednesday", "value": "wed"},
                                                {"label": "Thursday", "value": "thu"},
                                                {"label": "Friday", "value": "fri"},
                                                {"label": "Saturday", "value": "sat"},
                                                {"label": "Sunday", "value": "sun"},
                                            ],
                                            value="mon",
                                        ),
                                    ],
                                ),

                                html.Div(
                                    className="padding-top-bot",
                                    children=[
                                        html.H6("Select a testing date"),
                                        dcc.DatePickerSingle(
                                        id='date-testing',
                                        clearable=True,
                                        min_date_allowed=dt(2016, 1, 1),
                                        # max_date_allowed=dt(2019, 12, 31),
                                        placeholder='Select a date',
                                        display_format='DD/MM/YYYY'
                                        # date=dt(2018, 1, 9)
                                        )
                                    ],
                                ),

                                html.Div(
                                    className="padding-top-bot",
                                    children=[
                                        html.H6("Grouping interval"),
                                        dcc.RadioItems(
                                            id="grouping-interval",
                                            options=[
                                                {"label": "15 Minutes", "value": "15Min"},
                                                {"label": "30 Minutes", "value": "30Min"},
                                            ],
                                            value="15Min",
                                            labelStyle={
                                                "display": "inline-block",
                                                "padding": "2px 12px 2px 0px",
                                            },
                                        ),
                                    ],
                                ),

                                html.Div(
                                    className="padding-top-bot",
                                    children=[
                                        html.H6("Axis type"),
                                        dcc.RadioItems(
                                            id="axis-type",
                                            options=[
                                                # {"label": "Linear", "value": "linear"},
                                                {"label": "Log", "value": "log"},
                                            ],
                                            value="log",
                                            labelStyle={
                                                "display": "inline-block",
                                                "padding": "2px 12px 2px 0px",
                                            },
                                        ),
                                    ],
                                ),
                                
                                # Contact info
                                html.H6(" ", style={'padding': 16}),
                                html.Div(
                                    children=[
                                        dcc.Markdown('Created by Z. Wang'),
                                        dcc.Markdown('Email: zeliang@best.energy'),
                                        dcc.Markdown('Find out more about the app on'),
                                        html.A('Best.Energy', href='https://best.energy/', target='_blank'),
                                    ],
                                    style={'marginLeft': 0, 'marginRight': 0, 'marginTop': 10, 'marginBottom': 10,
                                            'padding': '0px 0px 0px 8px', 'align': 'left'}
                                ),
                            ],
                        )
                    ],
                ),
                # Graph
                html.Div(
                    className="nine columns card-left",
                    children=[
                        html.Div(
                            className="bg-white",
                            children=[
                                html.H5("Percentile ranges"),
                                dcc.Graph(id="plot-percentiles"),
                            ],
                        ),

                        html.Div(
                            className="bg-white",
                            children=[
                                html.H5("T-test"),
                                dcc.Graph(id="plot-pvalue"), 
                                html.Div(
                                    children=[
                                        # html.P(
                                        #     "Select the p-value threshold to which the null hypothesis can be rejected:"
                                        # ),
                                        dcc.Slider(
                                            id="pvalue-threshold",
                                            marks={v: '{}'.format(logscale_val[i]) for i, v in enumerate(range(-12,1,3))},
                                            min=-12,
                                            max=0,
                                            value=-3,
                                            step=0.1
                                        ), 
                                    ],
                                    # position a Div on center of page ('margin': '0 auto 0 auto')
                                    style = {'width': '80%', 'margin': '0 auto 0 auto', 'padding': '20px 0px 0px 0px'}
                                ),

                            ],
                        ),

                    ],
                ),

                # Error
                dcc.Store(id="error", storage_type="memory"),
                dcc.Store(id="error2", storage_type="memory"),
            ],
        ),
    ]
)

########################################################################################
# Callback to return UUID options for the specified case study
########################################################################################
@app.callback(
    [Output("eniscope-uuid", "options"),
     Output("eniscope-uuid", "value")],
    [Input("case-study", "value")],
)
# Update Eniscope UUID Dropdown
def update_uuid(case_study):
    options = []

    for uuid in uuid_all[case_study]:
        options.append({"label": uuid, "value": uuid})

    options.sort(key=lambda item: item["label"])

    # return the first value by default
    value = options[0]["value"] if options else None

    return options, value

########################################################################################
# Callback to return field-opt for the specified grouping interval
########################################################################################
@app.callback(
    [
        Output("field-opt", "options"),
        Output("field-opt", "value")
        ],
    [Input("grouping-interval", "value")],
)
# Update Eniscope UUID Dropdown
def update_field(interval_value):
    options = []

    for field in field_all[interval_value]:
        options.append({"label": field, "value": field})

    value = 'P' if options else None

    return options, value

########################################################################################
# Callback to plot percentile bands and testing data
########################################################################################

@app.callback(
    [
        Output("error", "data"),
        Output("error-message", "children"),
        Output("plot-percentiles", "figure")
    ],
    [
        # Input("confirm-btn", "submit_n_clicks"),
        
        Input("eniscope-uuid", "value"),
        Input("grouping-interval", "value"),
        Input("field-opt", "value"),
        Input("year-dropdown", "value"),
        Input("day-dropdown", "value"),
        Input("date-testing", "date")
    ]
)


def update_percentile(UUID, interval_val, field_val, year, day_val, date_testing):
    error_status = False
    error_message = None

    #
    if field_val in ['P', 'P1', 'P2', 'P3']:
        tick_suffix = 'kW'
        title_y = 'Power [kW]'
    elif field_val in ['E', 'E1', 'E2', 'E3']:
        tick_suffix = 'kWh'
        title_y = 'Energy [kWh]'
    else:
        tick_suffix = ' '
        title_y = ' '
    
    #
    if interval_val == '15Min':
        t = time_idx1
    elif interval_val == '30Min':
        t = time_idx1[::2]
    else:
        t = []

    #
    if None in (UUID, interval_val, field_val, year, day_val):    
        return error_status, error_message, {}

    else:
        data_stats = []
        try:
            # percentile bands
            for idx, perc_rng in enumerate(['0~100th','5~95th','10~90th', '15~85th', '20~80th']):
                lower_bound = perc_rng[:-2].split('~')[0]
                upper_bound = perc_rng[:-2].split('~')[1]
                trace = {
                    # 'x': np.arange(len(t)),
                    'x': t,
                    'y': df.loc[
                                    (df['year'] == int(year)) &
                                    (df['uuid'] == UUID) & 
                                    (df['day'] == day_val) & 
                                    (df['interval'] == interval_val) &   # 
                                    (df['field'] == field_val) & 
                                    (df['stats_group'] == perc_rng) &
                                    (df['stats'] == f'PERCENTILE_{lower_bound}'),
                                    t
                                ].values[0,:],
                    'hoverinfo': 'text',#'text+x',
                    'showlegend': False,
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': '{}th'.format(lower_bound),
                    'line': {'color': colors[idx],
                            'width': 0.05}
                }
                data_stats.append(trace)

                trace = {
                    # 'x': np.arange(len(t)),
                    'x': t,
                    'y': df.loc[
                                    (df['year'] == int(year)) &
                                    (df['uuid'] == UUID) & 
                                    (df['day'] == day_val) & 
                                    (df['interval'] == interval_val) &   # 
                                    (df['field'] == field_val) & 
                                    (df['stats_group'] == perc_rng) & 
                                    (df['stats'] == f'PERCENTILE_{upper_bound}'),
                                    t
                                ].values[0,:],
                    'hoverinfo': 'text',#'text+x',
                    'type': 'scatter',
                    'fill': 'tonexty',
                    'mode': 'lines',
                    'name': perc_rng,
                    'line': {'color': colors[idx],
                            'width': 0.2}
                }
                data_stats.append(trace)

            # mean
            trace = {
                # 'x': np.arange(len(t)),
                'x': t,
                'y': df.loc[
                                (df['year'] == int(year)) &
                                (df['uuid'] == UUID) & 
                                (df['day'] == day_val) & 
                                (df['interval'] == interval_val) &   # 
                                (df['field'] == field_val) & 
                                (df['stats'] == 'MEAN'), 
                                t
                            ].values[0,:],
                'hoverinfo': 'text',#'text+x',
                'type': 'scatter',
                'mode': 'lines',
                # 'showlegend': False,
                'name': 'Mean',
                'line': {'color': 'rgb(33, 33, 33)', 'width': 1}
            }
            data_stats.append(trace)

            # Add hover text
            for trace_ in data_stats:
                # trace['text'] = ['{:.2f}°C'.format(y) for y in trace['y']]
                hover_inputs = zip(trace_['x'], trace_['y'])
                trace_['text'] = [hovertext(x, y, tick_suffix)
                                for (x, y) in hover_inputs]

            annotation = [
                    {
                        "yanchor": "top",
                        "xref": "paper",
                        "xanchor": "right",
                        "yref": "paper",
                        "text": "Created by Z. Wang @ Best.Energy",
                        "y": 0.115,
                        "x": 1,
                        'align': 'right',
                        # "ay": -40,
                        # "ax": 0,
                        "showarrow": False,
                        'font': {
                            'color': '#DCDCDC',#'#A9A9A9',#'#d3d3d3',
                            'size': 9
                        }
                    }
                    ]

            if date_testing is not None:  # if testing date is selected
                # convert the selected date string to datetime object
                date_testing = dt.strptime(date_testing, '%Y-%m-%d')  
                date_str = date_testing.strftime('%Y%m%d')
                trace_testing = {
                    # 'x': np.arange(len(t)),
                    'x': t,
                    'y': df_testing.loc[
                                    (df_testing['date'] == int(date_str)) &
                                    (df_testing['uuid'] == UUID) & 
                                    (df_testing['field'] == field_val) & 
                                    (df_testing['interval'] == interval_val),
                                    # (df_testing['stats'] == 'MEAN'), 
                                    t
                                ].values[0,:],
                    'hoverinfo': 'text', #'text+x',
                    'type': 'scatter',
                    'mode': 'markers',
                    # 'showlegend': False,
                    'name': 'Mean (testing)',
                    'line': {'color': 'DarkSlateGrey', 'width': 1}
                }

                # Add hover text
                # trace['text'] = ['{:.2f}°C'.format(y) for y in trace['y']]
                hover_testing = zip(trace_testing['x'], trace_testing['y'])
                trace_testing['text'] = [hovertext(x, y, tick_suffix)
                                for (x, y) in hover_testing]

                data_stats.append(trace_testing)

        # capture Exception errors
        except Exception as e:  
            error_message = html.Div(
                className="alert",
                children=["The chosen statistical results have not been found in the input csv files! ExceptionError:{}".format(e)],
            )
            error_status = True

        #
        figure = go.Figure(
            data = data_stats,
            layout = go.Layout(
                # title = '',
                # showlegend=False,
                legend=go.layout.Legend(
                    orientation= 'h',
                    x=0.5,
                    y=-0.25,
                    xanchor='center'
                ),
                margin = go.layout.Margin(l=50, r=10, t=10, b=50),
                hovermode= 'closest',
                yaxis= {
                    # 'title': ' ',
                    'title': title_y,
                    'zeroline': False,
                },
                xaxis = {
                    'tickvals': ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00',
                                '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00',
                                '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'],
                    'showgrid': True,
                    'title': 'Time (UTC)',

                }
            )
        )

        return error_status, error_message, figure


########################################################################################
# Callback to plot the T-test results
########################################################################################
@app.callback(
    [
        Output("error2", "data"),
        Output("error-message2", "children"),
        Output("plot-pvalue", "figure")
    ],
    [
        Input("date-testing", "date"),
        Input("pvalue-threshold", "value"),
        Input("field-opt", "value"),
        Input("axis-type", "value"),
        Input("grouping-interval", "value"),
        Input("year-dropdown", "value"),
        Input("day-dropdown", "value")
    ],
    [
        State("case-study", "value"),
        State("eniscope-uuid", "value")
    ]
)
#
def update_ttest_plot(date_testing, threshold, field_val, axis_type, interval_val, year, day_val, case, UUID):

    error_status = False
    error_message = None 

    #
    if interval_val == '15Min':
        t = time_idx1
    elif interval_val == '30Min':
        t = time_idx1[::2]
    else:
        t = []


    if None in (date_testing, case, year, UUID, day_val, field_val, interval_val):
        return error_status, error_message, {}

    else:
        ORG_FOLDER = DIR_MAPPING[case]  #
        CONTAINER = "ENISCOPE_ARCHIVE_"+year  # choose which years data
        DEV_FOLDER = "preprocessed/" + UUID[:12]
        
        TRAINING_DIR = os.path.join(DATASETS_FOLDER, CONTAINER, ORG_FOLDER, DEV_FOLDER,"combined")
        training_data = UUID + "-" + year +"-" + day_val + "-1Min.csv"

        date_testing = dt.strptime(date_testing, '%Y-%m-%d')  # convert string to datetime object
        testing_date = date_testing.strftime('%Y%m%d')  # convert datetime to string
        CONTAINER_TESTING = "ENISCOPE_ARCHIVE_"+date_testing.strftime('%Y')  # choose which years data
        TESTING_DIR = os.path.join(DATASETS_FOLDER,CONTAINER_TESTING,ORG_FOLDER, DEV_FOLDER)

        if UUID == '001ec08959bf0004':  # for BEST case study
            delta = date_testing - dt(2019, 11,22) # No rackspace data available after 2019/11/21 for BEST case study
            if delta.days < 0:
                testing_data= UUID+"-"+testing_date+"-1Min.csv"
            else:
                testing_data= UUID+"-"+testing_date+"-1Min-analytics.csv"  # testing data from analytics
        else:  # for other case studies
            testing_data= UUID+"-"+testing_date+"-1Min.csv"

        data_pvals = []

        try:
            if not os.path.exists(TRAINING_DIR + "/" + training_data):
                error_message = html.Div(
                    className="alert",
                    children=["Training data: {} cannot be found!".format(training_data)],
                )
                error_status = True

            elif not os.path.exists(TESTING_DIR + "/" + testing_data):
                error_message = html.Div(
                    className="alert",
                    children=["Testing data: {} cannot be found!".format(testing_data)],
                )
                error_status = True

            else:

                # p value is only calculated for power values
                if field_val in ['P','P1','P2','P3']:
                    param_training = resample_params(TRAINING_DIR + "/" + training_data, interval_val, field_val)
                    param_testing = resample_params(TESTING_DIR + "/" + testing_data, interval_val, field_val)
                    p_vals = calc_ttest(param_training, param_testing)
                else:
                    p_vals = []

                #######################################################################################
                # t-test
                #######################################################################################

                for j, pval in enumerate(p_vals):
                    if pval <= 10**(threshold):   
                        c="red"
                    else:
                        c="blue"

                    new_trace = dict(
                        x=[t[j]],
                        y=[pval],
                        # name=str(well_id),
                        mode="lines+markers",
                        hoverinfo="x+y", # "x+y+name"
                        marker=dict(
                            symbol="hexagram-open", line={"width": 1.2}, color=c
                        ),
                        line=dict(shape="spline"),
                        # showlegend=True,
                    )
                    data_pvals.append(new_trace)

        except Exception as e:  # capture all Exception errors 
            error_message = html.Div(
                className="alert",
                children=["No record has been found in the input csv file!\n{}".format(e)],
            )
            error_status = True

        #######################################################################################
        # p value figure
        #######################################################################################
        if len(data_pvals) == 0:
            figure_ttest = {}
        else:
            figure_ttest = go.Figure(
                data = data_pvals,
                layout = go.Layout(
                    # title = '',
                    showlegend=False,
                    legend=go.layout.Legend(
                        orientation= 'h',
                        x=0.5,
                        y=-0.2,
                        xanchor='center'
                    ),
                    margin = go.layout.Margin(l=50, r=10, t=10, b=50),
                    hovermode= 'closest',
                    yaxis= {
                        # 'title': 'Probability of alternate hypothesis occured by change (p-value)',
                        'title': 'p-value',
                        # 'zeroline': False,
                        'type': axis_type
                    },
                    xaxis = {
                        'tickvals': ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00',
                                    '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00',
                                    '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'],
                        # 'zeroline': False,
                        'title': 'Time (UTC)',

                    }
                )
            )

        return error_status, error_message, figure_ttest

#######################################################################################
# Run server
#######################################################################################
if __name__ == "__main__":
    app.run_server(debug=True)
