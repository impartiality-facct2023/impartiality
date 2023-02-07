import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline
from scipy.interpolate import griddata
from scipy.optimize import minimize


# Very slow for many datapoints.  Fastest for many costs, most readable
def is_pareto_efficient_dumb(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any(costs[:i]>c, axis=1)) and np.all(np.any(costs[i+1:]>c, axis=1))
    return is_efficient


# Fairly fast for many datapoints, less fast for many costs, somewhat readable
def is_pareto_efficient_simple(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient


# Faster than is_pareto_efficient_simple, but less readable.
def is_pareto_efficient(costs, return_mask = True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient
    
    
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
    
    x1 = np.array(df1['achieved budget'])
    y1 = np.array(df1['fairness disparity gaps'])
    z1 = np.array(df1['number answered'])
    x2 = np.array(df2['achieved budget'])
    y2 = np.array(df2['fairness disparity gaps'])
    z2 = np.array(df2['number answered'])
    
    # concatenate the results
    x = np.concatenate((x1, x2), axis = 0)
    y = np.concatenate((y1, y2), axis = 0)
    z = np.concatenate((z1, z2), axis = 0)
    # stack
    results = np.stack((x, y, z), axis = -1)
    
    return results

def graph_data1(points, dataset):
    # 2d scatter plot with colour
    fig = px.scatter(x=points[:,0], y=points[:,2],color=points[:,1], color_continuous_scale='rainbow')
    fig.update_xaxes(title='Epsilon Budget Achieved')
    fig.update_yaxes(title='Number Answered')
    fig.update_layout(font=dict(
                    family="Times New Roman",
                    size=26
                ))
    
    fig.update_traces(marker_size=10)
    fig.layout.coloraxis.colorbar.title = 'Max Fair Violation'
    fig.update_coloraxes(colorbar_title_side='right')
    #fig.write_image("./figures/"+dataset+"_pareto2d.pdf")
    #time.sleep(2)
    #fig.write_image("./figures/"+dataset+"_pareto2d.pdf")

    fig.show()
    
    
def graph_data2(all_points, pf, dataset):
    
    fig = px.scatter_3d(x=all_points[:,0], y=all_points[:,1], z=all_points[:,2], color=all_points[:,2], color_continuous_scale='viridis')
    fig.update_layout(scene = dict(
                    xaxis_title='Epsilon Budget Achieved',
                    yaxis_title='Max Fair Violation',
                    zaxis_title='Number Answered'
                ))
    fig.add_scatter3d(x=pf[:,0], y=pf[:,1], z=pf[:,2], mode='markers', 
                  marker=dict(size=10,  symbol = 'diamond'))
    fig.show()
    
    
def graph_data_2var(all_points, axis_names, dataset):
    fig = px.scatter(x=all_points[:,0], y=all_points[:,1], z=all_points[:,2])
    fig.update_layout(scene = dict(
                    xaxis_title=axis_names[0],
                    yaxis_title=axis_names[1]
                ))
    fig.add_scatter3d(x=pf[:,0], y=pf[:,1], z=pf[:,2], mode='markers', 
                  marker=dict(size=10,  symbol = 'diamond'))
    fig.show()
    
    
def graph_surface(points, dataset):
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)

    X,Y = np.meshgrid(xi,yi)
    Z = griddata((x,y),z,(X,Y), method='linear')

    fig = go.Figure(data = [go.Surface(x=xi,y=yi,z=Z,colorscale='Viridis')])
    fig.update_layout(scene_camera_eye=dict(x=-1.7, y=2.0, z=0.75),
                      width=600, height=600,
                    # margin=dict(l=100, r=100, b=100, t=100),
                    scene=dict(
                    xaxis_title='Epsilon Budget Achieved',
                    yaxis_title='Max Fair Violation',
                    zaxis_title='Number Answered'),
                    xaxis={'automargin': True},
                    yaxis={'automargin': True})
    fig.update_traces(showscale=False)
    # fig.update_traces(colorbar_len=0.7)
    #fig.show()
    fig.write_image("./figures/"+dataset+"_pareto_surface.pdf")

def graph_3d_scatter(points, dataset):
    # 3d scatter plot
    fig = px.scatter_3d(x=points[:,0], y=points[:,1], z=points[:,2], color=points[:,2], color_continuous_scale='rainbow')
    fig.update_layout(scene = dict(
                    xaxis_title='Epsilon Budget Achieved',
                    yaxis_title='Max Fair Violation',
                    zaxis_title='Number Answered'
                ))
    fig.show()
    
    
def main():
    # this file graphs the pareto frontier of queries, with different options
    # options = {"2d scatter 2 var", "2d scatter 3 var", "surface", "3d scatter"}
    option = "surface"
    # variables = {"privacy", "fairness", "queries"}
    variables = {"privacy", "fairness"}
    datasets = [("celeba", "150"), ("colormnist", "200"), ("fairface", "50"), ("utkface", "100"), ("chexpert", "50")]
    #datasets = [("chexpert", "50")]
    for dataset, num in datasets:
        data = load_data(dataset, num)
        if option == "2d scatter 2 var":
            # graph selected two variables
            data_loss = np.zeros(shape=(data.shape[0], 2))
            index = 0
            axis_names = []
            for var in variables:
                if var == "privacy":
                    data_loss[:, index] = data[:, 0]
                    axis_names.append("Epsilon Budget Achieved")
                elif var == "fairness":
                    data_loss[:, index] = data[:, 1]
                    axis_names.append("Max Fair Violation")
                elif var == "queries":
                    data_loss[:, index] = data[:, 2]
                    axis_names.append("Number Answered")
                index += 1
        else:
            # graph all three variables
            # need to make number of queries answered into a loss
            data_loss = data.copy()
            data_loss[:, 2] = -data_loss[:, 2]
            indices = is_pareto_efficient(data_loss, False)
            points = data[indices, :]
            if option == "2d scatter 3 var":
                #print(point)
                graph_data1(points, dataset)
            elif option == "surface":
                graph_surface(points, dataset)
            elif option == "3d scatter":
                graph_3d_scatter(points, dataset)
            
    
    
if __name__ == "__main__":
    main()