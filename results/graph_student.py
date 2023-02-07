import pdb
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline
import scipy
from scipy.interpolate import griddata
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
from plotly.graph_objects import Surface


class TooFewDataException(Exception):
    pass

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
    
def load_generic_data(path):
    df = pd.read_csv(path)
    df = df.rename(columns={"fairness gaps": "fairness gap", "accuracy": "student accuracy", "achieved_budget": "achieved epsilon", "achieved_fairgap": "fairness gap"})
    return df

def load_data(dataset, method):
    '''
    loads the achieved fairness violation, achieved priavcy budget, and number of queries answered
    from data file.
    :param: The dataset of the results
    :return: An (n points, n metrics) array
    '''
    path = "./results/"
    csv_path = path+ f"{dataset}_{method}_students.csv"
    df = pd.read_csv(csv_path)

    return df

def load_data_baseline(dataset, method, processing):
    path = "./results/"
    csv_path = path+ f"baselines/{dataset}_{method}_{processing}.csv"
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"fairness gaps": "fairness gap", "accuracy": "student accuracy", "achieved_budget": "achieved epsilon", "achieved_fairgap": "fairness gap"})

    return df

def preprocess(dataset, df, bagging = True):
    x = np.array(df['achieved epsilon'])
    y = np.array(df['fairness gap'])
    z = np.array(df['student accuracy'])
    if dataset == 'chexpert':
        z = np.array(df['student auc'])
    c = np.array(df['coverage'])
    data = np.stack((x, y, z, c), axis = -1)
    if bagging:
        x1 = pd.qcut(x, 10, labels=False, duplicates='drop')
        y1 = pd.qcut(y, 10, labels=False, duplicates='drop')
        z1 = pd.qcut(z, 10, labels=False, duplicates='drop')
        c1 = pd.qcut(c, 10, labels=False, duplicates='drop')
        data_loss = np.stack((x1, y1, -z1, -c1), axis = -1)
    else:
        data_loss = np.stack((x, y, -z, -c), axis = -1)

    return data, data_loss

def get_pretty_name(m, p, dataset=None, args=None):
    if dataset is not None and "multidataset" in args.method:
        if dataset == "chexpert":
            return "CheXpert"
        elif dataset == "colormnist":
            return "ColorMNIST"
        elif dataset == "fairface":
            return "FairFace"
        elif dataset == "utkface":
            return "UTKFace"
        elif dataset == "celeba":
            return "CelebA"
        else:
            raise NotImplementedError
        
    if m == "pate_pre" and p is None:
        return "FairPATE"
    elif m == "pate" and p is None:
        return "FairPATE Without Post-processor"
    elif m == "dpsgd_pre" and p == "inprocessing":
        return "FairDP-SGD"
    elif m == "dpsgd" and p == "inprocessing":
        return "FairDP-SGD Without Post-processor"
    elif m == "vanilla_pate" and p == "preprocessing":
        return "PATE-Pre Without Post-processor"
    elif m == "vanilla_pate_pre" and p == "preprocessing":
        return "PATE-Pre"
    elif m == "vanilla_pate" and p == "inprocessing":
        return "PATE-In Without Post-processor"
    elif m=="vanilla_pate_pre" and p =="inprocessing":
        return "PATE-In"
    elif args.input is not None:
        return args.input[args.fileid].split("/")[-1].split(".")[0]
    else:
        raise NotImplementedError

def graph_surface(points, dataset, method, points2 = None, processing=None, pareto_indices=None, non_pareto_indices=None, use_plotly=True, use_matplotlib=False, get_traces=False, add_scatter=True, add_annotation=True, initially_hide_scatter=False, colormap='Viridis', uid=None, is_paperplot=False, separate_colorbars=False, scene_camera_eye=dict(x=-1.7, y=2.0, z=0.75), args=None, cmin=0.65):

    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
    c = points[:,3]
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    X,Y = np.meshgrid(xi,yi)
    Z = griddata((x,y),z,(X,Y), method='linear')
    C = griddata((x,y),c,(X,Y), method='linear')

    if all([pareto_indices is not None, non_pareto_indices is not None]):
        x_pareto = points[pareto_indices][:,0]
        y_pareto = points[pareto_indices][:,1]
        z_pareto = points[pareto_indices][:,2]
        c_pareto = points[pareto_indices][:,3]
        xi_pareto = np.linspace(x_pareto.min(), x_pareto.max(), 100)
        yi_pareto = np.linspace(y_pareto.min(), y_pareto.max(), 100)
        X_pareto,Y_pareto = np.meshgrid(xi_pareto,yi_pareto)
        Z_pareto = griddata((x_pareto,y_pareto),z_pareto,(X_pareto,Y_pareto), method='linear')
        C_pareto = griddata((x_pareto,y_pareto),c_pareto,(X_pareto,Y_pareto), method='linear')

        skip_non_pareto = False
        if len(non_pareto_indices) == 0:
            print("Empty Non-Pareto indices")
            skip_non_pareto = True
        
        if use_plotly:
            textz = np.array([[f'x: {X_pareto[i, j]:0.2f}<br> y: {Y_pareto[i, j]:0.2f}<br>z: {Z_pareto[i, j]:0.2f}<br>c: {C_pareto[i, j]:0.2f}'
                        for j in range(Z_pareto.shape[1])] 
                            for i in range(Z_pareto.shape[0])]) 

            colorbar_dict = dict(x=-0.2, thickness=27, len=0.5)
            
            if separate_colorbars:
                if uid is None:
                    pass
                else:
                    colorbar_dict = colorbar_dict | dict(x=-0.1 +uid*-0.05*(10/27), thickness=10)
                    if uid == 0:
                        colorbar_dict = {"ticks": "outside", "title": dict(text="Coverage", side="right"), **colorbar_dict}
                    else:
                        colorbar_dict  = {"ticks": "outside", "title": dict(text="", side="right"), "ticks": "", "showticklabels": True, "tickmode": "array", "tickvals": [], **colorbar_dict}

            else:
                if uid == 0:
                    colorbar_dict = colorbar_dict | dict(x=-0.2, title=dict(text="Coverage", side="top"))
                else:
                    colorbar_dict = colorbar_dict | dict(x=-0.2)
                    
            plotly_traces = [
                go.Surface(x=xi_pareto,y=yi_pareto,z=Z_pareto,
                    text=textz,
                    surfacecolor=C_pareto,
                    meta=C_pareto,
                    colorscale=colormap,
                    contours=dict(x=dict(show=False, color="black"), y=dict(show=False, color="black"), z=dict(show=False, color="black")),
                    cmax=1.0, 
                    cmin=cmin,
                    # hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<br>c: %{meta:.2f}",
                    hoverinfo='text',
                    opacity=1,
                    name=get_pretty_name(method, processing, dataset=dataset, args=args) if get_traces else "Pareto Surface",
                    showscale=False if args.hide_colorbar else True if separate_colorbars else False if uid > 0 else True,
                    showlegend=False if args.hide_legend else True,
                    colorbar=colorbar_dict,
                    uid=uid
                    ),
                # go.Scatter3d(x=X_pareto, y=Y_pareto, z=Z_pareto, mode='lines', name="Wireframe", line=dict(color="black")),
            ]

            convert_to_pgfformat(f"./figures/{dataset}_{method}_{processing}_pareto_surface.csv", X_pareto, Y_pareto, Z_pareto, C_pareto)

            if add_scatter:
                plotly_traces += [
                    go.Scatter3d(
                        x=x[pareto_indices], 
                        y=y[pareto_indices], 
                        z=z[pareto_indices], 
                        meta=c[pareto_indices], 
                        mode='markers+text', 
                        marker_symbol="square", 
                        showlegend=False if is_paperplot else True,
                        name=get_pretty_name(method, processing, dataset=dataset, args=args) + " Pareto",
                        texttemplate="(%{x:.2f}, %{y:.2f}, %{z:.2f}, %{meta:.2f})" if add_annotation else "",
                        hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<br>c: %{meta:.2f}",
                        marker=dict(
                            size=8,
                            line=dict(color='black',width=400),
                            color=c[pareto_indices],            
                            cmax=1.0, 
                            cmin=cmin,
                            colorscale=colormap,   
                            opacity=1.0,
                            ),
                            visible=True if is_paperplot else 'legendonly' if not initially_hide_scatter else True
                            ),
                ]
            
            if not skip_non_pareto:
                plotly_traces += [go.Scatter3d(
                        x=x[non_pareto_indices], 
                        y=y[non_pareto_indices], 
                        z=z[non_pareto_indices], 
                        meta=c[non_pareto_indices], 
                        mode='markers+text',  
                        marker_symbol="square",  
                        name=get_pretty_name(method, processing, dataset=dataset, args=args) +" Non-Pareto",
                        texttemplate="(%{x:.2f}, %{y:.2f}, %{z:.2f}, %{meta:.2f})" if add_annotation else "",
                         marker=dict(
                            size=8,
                            line=dict(color='black',width=0),
                            color=c[non_pareto_indices],            
                            cmax=1.0, 
                            cmin=cmin,
                            colorscale=colormap,   
                            opacity=1.0,), 
                            visible=False if is_paperplot else 'legendonly'
                    )]
    else:
        fig = go.Figure(data = [
            go.Surface(x=xi,y=yi,z=Z,
            surfacecolor=C,
            colorscale='Viridis',
            cmax=1.0, 
            cmin=cmin,
            hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<br>c: %{surfacecolor:.2f}",
            opacity=0.7
            )
        ])

    if use_plotly:
        if get_traces:
            return plotly_traces
        else:
            fig = go.Figure(data = plotly_traces)
            fig.update_layout(scene_camera_eye=scene_camera_eye,
                        # margin=dict(l=100, r=100, b=100, t=100),
                        scene=dict(
                        xaxis_title='Epsilon Budget Achieved',
                        yaxis_title='Max Fairness Violation',
                        zaxis_title='Accuracy',
                        #xaxis = dict(nticks=5, range=[1,10],),
                        #yaxis = dict(nticks=3, range=[0,0.25],),
                        #zaxis = dict(nticks=5, range=[65,82],)
                        ),
                        xaxis={'automargin': True},
                        yaxis={'automargin': True},
                        font_size=17,
                        font_family="Libertine"
                        )

            if not args.no_box:
                fig.update_layout(width=args.width, height=args.height)


        # fig.update_traces(selector=dict(type="surface"), colorbar_len=0.7, colorbar_title_text="Coverage")
        fig.show()
   
        fig.write_image(f"./figures/{dataset}_{method}_{processing}_pareto_surface.pdf")
        fig.write_html(f"./figuresWebpage/static/{dataset.replace(',', '-')}_{method}_{processing}_pareto_surface.html", include_plotlyjs=False, auto_open=False, full_html=False, include_mathjax="cdn", default_width="1000px", default_height="800px")


    elif use_matplotlib:
        colormap_name = "cubehelix"
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111, projection='3d', computed_zorder=False, box_aspect=(3,3, 2))
        ax.set_aspect('auto', adjustable='box')
        norm = mpl.colors.Normalize(vmin=0.75, vmax=1)
        surf = ax.plot_surface(
            X_pareto, Y_pareto, Z_pareto, rstride=1, cstride=1, 
            cmap=colormap_name, facecolors=mpl.colormaps[colormap_name](norm(C_pareto)), norm=norm,
            antialiased=True, shade=False, zorder=1)

        ax.plot_wireframe(X_pareto, Y_pareto, Z_pareto, color='black', linewidth=0.2)

        ax.scatter(x[pareto_indices], y[pareto_indices], z[pareto_indices], c=c[pareto_indices], edgecolors="black", norm=norm, cmap=colormap_name, zorder=4)

        ax.scatter(x[non_pareto_indices], y[non_pareto_indices], z[non_pareto_indices], c=c[non_pareto_indices], edgecolors=None, norm=norm, cmap=colormap_name, zorder=4)

        for x_val, y_val, z_val, c_val in zip(x[pareto_indices], y[pareto_indices], z[pareto_indices], c[pareto_indices]):
            label = f"({x_val:0.2f}, {y_val:0.2f}, {z_val:0.2f}, {c_val:0.2f})"
            ax.text(x_val, y_val, z_val, label, None, fontsize=4, ha='center', va="bottom")

        ax.w_xaxis.set_pane_color((0.847, 0.847, 0.847, 1.0))
        ax.w_yaxis.set_pane_color((0.847, 0.847, 0.847, 1.0))
        ax.w_zaxis.set_pane_color((0.847, 0.847, 0.847, 1.0))

        ax.elev = 25
        ax.azim = 134
        fig.colorbar(surf, shrink=0.5, aspect=5)
        fig.tight_layout()
        fig.savefig(f"./figures/{dataset}_{method}_{processing}_pareto_surface.pdf")
        plt.show(block=True)


def convert_to_pgfformat(file_path, X, Y, Z, C):
    with open(file_path, "w") as f:
        f.write("epsilon, violation, accuracy, coverage\n")
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                f.write(f"{X[i, j]}, {Y[i, j]}, {Z[i, j]}, {C[i, j]}\n") 
            f.write("\n")

def graph_3d_scatter(points, dataset, method):
    # 3d scatter plot
    fig = px.scatter_3d(x=points[:,0], y=points[:,1], z=points[:,2], color=points[:,3], color_continuous_scale='Viridis')
    fig.update_layout(scene_camera_eye=dict(x=-1.7, y=-2.0, z=0.75),
                      width=600, height=600,
                      scene = dict(
                    xaxis_title='Epsilon Budget Achieved',
                    yaxis_title='Max Fair Violation',
                    zaxis_title='Student Accuracy'
                ))
    
    fig.show()
    
    if method == 'comparison':
        fig.write_image("./figures/"+dataset+"_compare_pareto.pdf")
    elif method == 'dpsgd':
        fig.write_image("./figures/"+dataset+"_dpsgd_pareto.pdf")
    else:
        fig.write_image("./figures/"+dataset+"_student_pareto.pdf")
    
    
def create_pgfplot_scatter_data(data, path, pareto_indices, non_pareto_indices):
    pd.DataFrame(data[pareto_indices], columns=["epsilon", "violation", "accuracy", "coverage"]).to_csv(path+"_pareto.csv", index=False)
    if len(non_pareto_indices) > 0:
        pd.DataFrame(data[non_pareto_indices], columns=["epsilon", "violation", "accuracy", "coverage"]).to_csv(path+"_non_pareto.csv", index=False)
    else:
        pd.DataFrame([], columns=["epsilon", "violation", "accuracy", "coverage"]).to_csv(path+"_non_pareto.csv", index=False)


def main():

    def plot_one(dataset, method, processing, colormap, get_traces=False, uid=None, input_file=None,cmin=0.65):
        if input_file is not None:
            df = load_generic_data(input_file)
        elif processing is not None:
            df = load_data_baseline(dataset, method, processing)
        else:
            df = load_data(dataset, method)
                
        data, data_loss = preprocess(dataset, df, bagging)
        pareto_indices = is_pareto_efficient(data_loss, False)
        non_pareto_indices = np.array([i for i in range(data.shape[0]) if i not in pareto_indices])
        create_pgfplot_scatter_data(data, f"./figures/{dataset}_{method}_{processing}_scatter", pareto_indices, non_pareto_indices)
        # Return traces
        return graph_surface(data, dataset, method, processing=processing, pareto_indices=pareto_indices, non_pareto_indices=non_pareto_indices, get_traces=get_traces, colormap=colormap, add_annotation=args.annotate, initially_hide_scatter=args.initially_hide_scatter, separate_colorbars=args.separate_colorbars, uid=uid, is_paperplot=args.paperplot, args=args, cmin=args.colour_min)
        
    def plot_traces(traces, legend_y, scene_camera_eye=dict(x=-1.7, y=2.0, z=0.75), legend_location="top",y_min=0.67):
        fig = go.Figure(data = traces)
        if legend_location == "top":
            legend_dict = dict(
                            orientation="h",
                            # yanchor="top",
                            y=legend_y,
                            # xanchor="left",
                            # x=1.0
                        )
        elif legend_location == "right":
            legend_dict = dict(
                                orientation="v",
                                yanchor="top",
                                y=.7,
                                xanchor="left",
                                x=1
                            )
        else:
            raise NotImplementedError

        fig.update_layout(scene_camera_eye=scene_camera_eye,
                        margin=dict(l=100, r=100, b=100, t=100),
                        scene=dict(
                        xaxis_title='Epsilon Budget Achieved',
                        yaxis_title='Max Fairness Violation',
                        zaxis_title='Accuracy',
                        xaxis = dict(nticks=5, range=[1,10],),
                        yaxis = dict(nticks=4, range=[0,0.25],),
                        zaxis = dict(nticks=5, range=[y_min,82],)),
                        xaxis={'automargin': True},
                        yaxis={'automargin': True},
                        font_size=17,
                        font_family="Libertine",
                        legend_font_size=20,
                        )

        if not args.no_box:
                fig.update_layout(width=args.width, height=args.height, legend=legend_dict)

        fig.show()
        fig.write_image(f"./figures/{dataset.replace(',', '-')}_{method}_{processing}_pareto_surface.pdf")
        fig.write_html(f"./figuresWebpage/static/{dataset.replace(',', '-')}_{method}_{processing}_pareto_surface.html", include_plotlyjs=False, auto_open=False, full_html=False, include_mathjax="cdn", default_width="1000px", default_height="800px")


    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", help="ukrface, colormnist, chexpert, fairface", type=str)
    parser.add_argument("-m", "--method", dest="method", nargs='?', default="all", help="dpsgd, pate (this is fairPATE), vanilla_pate, pate_multidataset, all (default)", type=str)
    parser.add_argument("-p", "--processing", dest="processing", nargs='?', help="preporcessing, inprocessing, None (default)", default=None, type=str)
    parser.add_argument("-bg", "--bagging", nargs='?', dest="bagging", default=False, type=bool)
    parser.add_argument("-a", "--annotate", action='store_true', help="Annotate the scatter plots. You may want to initially hide all scatterplots with '-hs' as well.")
    parser.add_argument("-hs", "--initially_hide_scatter", action='store_false', help="Initially hide scatter plots")
    parser.add_argument("-pp", "--paperplot", action='store_true', help="Create paper plots?")
    parser.add_argument('-c', '--camera', action='store', dest='camera',
                    type=float, nargs='*', default=[-1.7, 2.0, 0.75],
                    help="Set camera x y z coordinate: -1.7 2.0 0.75 (default) ")
    parser.add_argument("-l", "--legend_location", type=str, dest="legend_location", default="top", help="Set legend location: right, top (default)")
    parser.add_argument("-w", "--width", type=int, default=900, help="Set figure width")
    parser.add_argument("-ht", "--height", type=int, default=1000, help="Set figure height")
    
    parser.add_argument("-sc", "--separate_colorbars", action='store_true', help="Show seperate colorbars?")
    parser.add_argument("-hl", "--hide_legend", action='store_true', help="hide legends?")
    parser.add_argument("-hc", "--hide_colorbar", action='store_true', help="hide colorbar?")
    parser.add_argument("-cp", "--colour_palette", action='store', dest='colour_palette',
                        default='Greys,Purples,Blues,Greens,Oranges,Reds,YlOrBr,YlOrRd,OrRd,PuRd,RdPu,BuPu,GnBu,PuBu,YlGnBu,PuBuGn,BuGn,YlGn',
                        help="Define colour palette")

    parser.add_argument("-nb", "--no_box", action='store_true', help="No box for figure? (adjustable width/height)")
    parser.add_argument("-i", "--input", nargs='+', default=None, help="Provide a list of input CSV paths (you still need to provide the dataset name) ", type=str)
    parser.add_argument("-ly", "--legend_y", default=0.8, dest= "lengend_y", type=float, help="Provide location for the legend")
    parser.add_argument("-ym", "--y_min", default=67, dest= "y_min", type=int, help="Set y axis min value")
    parser.add_argument('-cm', "--colour_min", default=0.65, dest="colour_min", type=float, help="Set min value of the colour bar.")
    args = parser.parse_args()

    method = args.method
    bagging = args.bagging
    processing = args.processing
    dataset = args.dataset
    legend_y = args.lengend_y
    print(legend_y)

    if len(args.camera) != 3:
        raise Exception("Camera should be given exactly 3 coordiantes")
    else:
        scene_camera_eye = dict(x=args.camera[0], y=args.camera[1], z=args.camera[2])

    colors = iter(args.colour_palette.split(","))
    traces = []
    uid = 0

    if args.input is not None:
        for idx, file_path in enumerate(args.input):
            try:
                print(file_path)
                args.fileid = idx
                trace = plot_one(dataset, None, processing, colormap=next(colors), get_traces=True, uid=uid, input_file=file_path, cmin=args.colour_min)
                traces += trace
                uid += 1
            except FileNotFoundError:
                print("Did not found")
                continue
            except scipy.spatial.qhull.QhullError:
                print("Has too few datapoints. Skipping")
                continue
        plot_traces(traces, scene_camera_eye=scene_camera_eye, legend_location=args.legend_location, legend_y=legend_y, y_min=args.y_min)
    elif method == "pate" or method == "pate_pre":
        plot_one(dataset, method, processing, "viridis",uid=uid)
    elif "," in method or  ((processing is not None) and ("," in processing)):
        methods = method.split(",")
        processings = processing.split(",")
        for m in methods:
            for p in processings:
                if (m == "pate" or m == "pate_pre"):
                    p = None
                try:
                    print(m, p)
                    trace = plot_one(dataset, m, p, colormap=next(colors), get_traces=True, uid=uid)
                    traces += trace
                    uid += 1
                except FileNotFoundError:
                    print("Did not found")
                    continue
                except scipy.spatial.qhull.QhullError:
                    print("Has too few datapoints. Skipping")
                    continue
            plot_traces(traces, legend_y=legend_y, scene_camera_eye=scene_camera_eye, legend_location=args.legend_location, y_min=args.y_min)

    elif method == "all":
        if "," in dataset:
            raise Exception("Single dataset only. Run with a *_multidataset method to provide a list of datasets.")
            

        for m in ['dpsgd', 'pate', 'vanilla_pate']:
            for p in [None, 'inprocessing', 'preprocessing']:
                if (m == "pate") and p is not None or (m != "pate" and p is None):
                    continue # pate is fairPATE, so processing is not meaningful (it should be None)
                try:
                    print(m, p)
                    trace = plot_one(dataset, m, p, colormap=next(colors), get_traces=True, uid=uid)
                    traces += trace
                    uid += 1
                except FileNotFoundError:
                    print("Did not found")
                    continue
                except scipy.spatial.qhull.QhullError:
                    print("Has too few datapoints. Skipping")
                    continue
        plot_traces(traces, legend_y=legend_y, scene_camera_eye=scene_camera_eye, legend_location=args.legend_location, y_min=args.y_min)

    elif method == "pate_multidataset":
        traces = []
        uid = 0
        datasets = args.dataset.split(",")
        for d in datasets:
            try:
                print(d)
                trace = plot_one(d, "pate", None, colormap=next(colors), get_traces=True, uid=uid)
                traces += trace
                uid += 1
            except FileNotFoundError:
                print("Did not found")
                continue
            except scipy.spatial.qhull.QhullError:
                print("Has too few datapoints. Skipping")
                continue
        plot_traces(traces, legend_y=legend_y, scene_camera_eye=scene_camera_eye, legend_location=args.legend_location, y_min=args.y_min)

    else:
        print("Undefined method specification")

if __name__ == "__main__":
    main()
