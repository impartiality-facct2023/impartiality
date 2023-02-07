import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline
from scipy.interpolate import griddata
from scipy.optimize import minimize
import time

def load_data(dataset, size):
    '''
    loads the achieved fairness violation, achieved priavcy budget, and number of queries answered
    from data file.
    :param: The dataset of the results
    :return: An (n points, n metrics) array
    '''
    path = "./results/"
    df1 = pd.read_csv(path+dataset+'_'+size+'_search_fig1.csv')
    df2 = pd.read_csv(path+dataset+'_'+size+'_search_fig2.csv')

    return df1, df2
    
def graph_fig1(dataset, df1):
    x = np.array(df1['threshold'])
    y = np.array(df1['max fairness violation'])
    z = np.array(df1['number answered'])
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)

    X,Y = np.meshgrid(xi,yi)
    Z = griddata((x,y),z,(X,Y), method='cubic')

    fig = go.Figure(data = [go.Surface(x=xi,y=yi,z=Z,colorscale='Viridis')])
    fig.update_layout(#autosize=True,
                    scene_camera_eye=dict(x=2.0, y=-2.0, z=0.75),
                    width=600, height=600,
                    # margin=dict(l=100, r=100, b=100, t=100),
                    scene=dict(
                    yaxis_title='Fairness Threshold',
                    xaxis_title='Consensus Threshold',
                    zaxis_title='Number Answered'))
    fig.update_traces(showscale=False)
    # fig.update_traces(colorbar_len=0.7)
    #fig.show()
    
    fig.write_image("./figures/"+dataset+"_fig1.pdf")
    
def graph_fig2(dataset, df2):
    x2 = np.array(df2['achieved budget'])
    y2 = np.array(df2['fairness disparity gaps'])
    z2 = np.array(df2['number answered'])
    
    fig = px.scatter(x=x2, y=y2,color = z2, color_continuous_scale='Viridis')
    fig.update_xaxes(title='Epsilon Budget Achieved')
    fig.update_yaxes(title='Max Fair Violation')

    fig.update_layout(
        font=dict(
            family="Times New Roman",
            size=26
        )
    )

    fig.update_traces(marker_size=10)
    fig.layout.coloraxis.colorbar.title = 'Number Answered'
    fig.update_coloraxes(colorbar_title_side='right')
    #fig.show()
    fig.write_image("./figures/"+dataset+"_fig2.pdf")
    
def main():
    #datasets = [("celeba", "150"), ("colormnist", "200"), ("fairface", "50"), ("utkface", "100")]
    datasets = [("chexpert", "50")]    
    for dataset, num in datasets:
        df1, df2 = load_data(dataset, num)
        graph_fig1(dataset, df1)
        graph_fig2(dataset, df2)
    
if __name__ == "__main__":
    main()