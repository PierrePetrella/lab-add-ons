import dataiku
from dataiku.customwebapp import *
#from dataiku.customwebapp import get_webapp_config

from dataiku.doctor.posttraining.model_information_handler import PredictionModelInformationHandler

from dash import html
import dash_bootstrap_components as dbc
from dash import dcc
from dash.dependencies import Input, Output

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

import numpy as np
import pandas as pd

import matplotlib



#def get_model_handler(model, version_id=None):
#    params = model.get_predictor(version_id).params
#    return PredictionModelInformationHandler(params.split_desc, params.core_params, params.model_folder,
#                                             params.model_folder)

#def get_original_model_handler(webapp_config):
#    fmi = webapp_config.get("trainedModelFullModelId")
#    if fmi is None:
#        model = dataiku.Model(webapp_config["modelId"])
#        version_id = webapp_config.get("versionId")
#        original_model_handler = get_model_handler(model, version_id)
#    else:
#        original_model_handler = PredictionModelInformationHandler.from_full_model_id(fmi)
#    return original_model_handler

webapp_config = get_webapp_config()
palette = '#D5D9D9', '#3075AE', '#ff7e0b'
#ave_data, target, weight, class_map = get_ave_data(webapp_config)
#ave_grouped = get_ave_grouped(ave_data, target, weight, class_map)
#features = [k for k in ave_grouped.keys()]

#model_handler = get_original_model_handler(webapp_config)
#predictor = model_handler.get_predictor()

#if not isinstance(predictor._clf, BaseGLM):
#    raise ValueError('GLM Summary is only available for GLMs')

FA = "https://use.fontawesome.com/releases/v5.12.1/css/all.css"

webapp_plugin_assets = os.path.join(
    get_webapp_resource(), "../webapps/avse/assets"
)
dash_webapp_assets = app.config.assets_folder
print(
    f"Copying Webapp assets from directory '{webapp_plugin_assets}' into directory '{dash_webapp_assets}'"
)
print (webapp_config)

client = dataiku.api_client()
project = client.get_default_project()

def get_clustering_model_details(webapp_config)
    trained_model_ids = webapp_config["trainedModelFullModelId"].split("-")
    analysis = project.get_analysis(trained_model_ids[2])
    ml_task = analysis.get_ml_task(trained_model_ids[3])
    return ml_task.get_trained_model_details(webapp_config["trainedModelFullModelId"])

# Extract clustering model data
trained_model = get_clustering_model_details(webapp_config)
cluster_data = trained_model.get_raw()['heatmap']
cluster_labels = cluster_data['cluster_labels']
var_importance = trained_model.get_raw()["perf"]["variables_importance"]
features = var_importance["variables"].copy()

# Extract cluster summaries
cluster_summary = {}
for cluster_idx, cluster_label in enumerate(cluster_labels):
    cluster_summary[cluster_label] = []
    for feature in features:
        if feature in cluster_data["num_names"]:
            feat_index = cluster_data["num_names"].index(feature)
            # Uncomment to divide all the cluster values by their average.
            #prop_val = cluster_data["cluster_num_averages"][cluster_idx][feat_index]/cluster_data["num_averages"][feat_index]
            prop_val = cluster_data["cluster_num_averages"][cluster_idx][feat_index]
            cluster_summary[cluster_label].append(prop_val)
        elif feature.split(":")[1] in cluster_data["cat_names"]:
            cat_method,feat_name,cat_val = feature.split(":")
            feat_index = cluster_data["cat_names"].index(feat_name)
            cat_index = cluster_data["levels"][feat_index].index(cat_val)
            cluster_summary[cluster_label].append(cluster_data["cluster_proportions"][cluster_idx][feat_index][cat_index])
        else:
            print ("Feature type not supported")
            break


colors = dict(matplotlib.colors.cnames.items())
hex_colors = tuple(colors.values())
fig = go.Figure()
for cluster_idx, cluster_label in enumerate(cluster_labels):
    fig.add_trace(go.Scatterpolar(
            r = cluster_summary[cluster_label],
            theta = features,
            mode = 'lines',
            name = cluster_label,
            line_color = hex_colors[cluster_idx + 40],))
# Uncomment to divide all the cluster values by their average.
#fig.add_trace(go.Scatterpolar(
#            r = [1]*len(features),
#            theta = features,
#            mode = 'lines',
#            name = "base",
#            line_color = "blue",))


app.layout = html.Div([html.H1("Radar charts of clusters"),
                       html.H3("cluster avg./general feature avg for numerical features"),
                       html.H3("Category proportion for categorical features"),
                       html.Div(
                          [html.Div(dcc.Graph(figure=fig), style={'display': 'inline-block'}),
                          ],style={'display': 'inline-block', 'text-align': 'justify'}),
                      html.H3(str(type(webapp_config))),
                      html.H3(str(webapp_config)),
                      ]
                     )


