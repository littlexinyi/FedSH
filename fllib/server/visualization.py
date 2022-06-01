import numpy as np


def vis_scalar(vis, figure_name, scalar_name, x, y):
    
    vis.line(X=np.array([x]), Y=np.array([y]), 
                name=scalar_name, win=figure_name, 
                update='append' if vis.win_exists(figure_name) else None,
                opts=dict(showlegend=False, xlabel='Communication rounds', ylabel=scalar_name)
                ) 

