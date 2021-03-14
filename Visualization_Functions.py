# Functions for visualizing pointclouds.
# ClassesNames = ['Bathtub', 'Bed', 'Chair', 'Desk', 'Dresser', 'Monitor', 'Night_Stand', 'Sofa', 'Table', 'Toilet']

def plotPC(data, Myclasses=['Bathtub', 'Bed', 'Chair', 'Desk', 'Dresser', 'Monitor', 'Night_Stand', 'Sofa', 'Table', 'Toilet']):   
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    x = data.pos
    y = data.y
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(projection='3d')
    ax.scatter3D(x[:,0], x[:,1], x[:,2])  #data.pos[num_nodes, num_dimensions] -> node position (x,y,z)
    ax.set(xlabel='x', ylabel='y', zlabel='z')
    axes = ax.axes
    max = torch.max(data.pos)
    min = torch.min(data.pos)
    max = np.maximum(torch.abs(max),torch.abs(min))
    axes.set_xlim(-max,max)
    axes.set_ylim(-max,max)
    axes.set_zlim(-max,max)
    print(Myclasses[y])
    plt.show()


def plotGraph3D(data, Myclasses=['Bathtub', 'Bed', 'Chair', 'Desk', 'Dresser', 'Monitor', 'Night_Stand', 'Sofa', 'Table', 'Toilet']):
    import plotly.graph_objects as go
    import numpy as np
    from numpy import allclose
    from plotly.offline import iplot
    from sklearn.metrics import confusion_matrix
    numNodes = data.pos.shape[0]
    if data.edge_index==None:
      print('Not a Graph. Use plotPC instead.')
      return
    numEdges = data.edge_index.shape[1]
    Xn=[]
    Yn=[]
    Zn=[]
    for k in range(numNodes):
      Xn+=[data.pos[k,0]]
      Yn+=[data.pos[k,1]]
      Zn+=[data.pos[k,2]]
    Xe=[]
    Ye=[]
    Ze=[]
    a = []
    b = []
    for e in range(numEdges):
      src = int(data.edge_index[0,e])
      tgt = int(data.edge_index[1,e])
      Xe+=[data.pos[src,0].item(), data.pos[tgt,0].item(),None]# x-coordinates of edge ends
      Ye+=[data.pos[src,1].item(), data.pos[tgt,1].item(),None]
      Ze+=[data.pos[src,2].item(), data.pos[tgt,2].item(),None]
      a.append(data.edge_index[0,e].item())
      b.append(data.edge_index[1,e].item())
    trace1=go.Scatter3d(x=Xe, y=Ye, z=Ze, mode='lines', line=dict(color='rgb(125,125,125)', width=1))
    trace2=go.Scatter3d(x=Xn, y=Yn, z=Zn, mode='markers', 
                   marker=dict(symbol='circle', size=2, colorscale='Viridis', 
                      line=dict(color='rgb(50,50,50)', width=0.25)))
    axis=dict(showbackground=False, showline=False, zeroline=False, showgrid=False, showticklabels=True)
    layout = go.Layout(
         autosize=True,
         #width=1000,height=1000,
         title=Myclasses[data.y],
         showlegend=False,
         scene=dict(aspectmode='data',xaxis=dict(axis),yaxis=dict(axis),zaxis=dict(axis))
        )

    cm = confusion_matrix(a,b)

    IsUnDirected = np.allclose(cm, cm.T, rtol=1e-08, atol=1e-08)
    if IsUnDirected:
      IsDirected = 'False'
    else:
      IsDirected = 'True'
    print("\nNodes:", numNodes, "\tEdges:", numEdges, "\tEdges/Node avg: {:.2f}".format(numEdges/numNodes))
    print("Is Directed: ", IsDirected, " (if False, edges defined both ways)")
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    iplot(fig)


def plotPC3D(data, Myclasses=['Bathtub', 'Bed', 'Chair', 'Desk', 'Dresser', 'Monitor', 'Night_Stand', 'Sofa', 'Table', 'Toilet']):
    import plotly.graph_objects as go
    from plotly.offline import iplot
    numNodes = data.pos.shape[0]
    Xn=[]
    Yn=[]
    Zn=[]
    for k in range(numNodes):
      Xn+=[data.pos[k,0]]
      Yn+=[data.pos[k,1]]
      Zn+=[data.pos[k,2]]
    trace2=go.Scatter3d(x=Xn, y=Yn, z=Zn, mode='markers', 
                   marker=dict(symbol='circle', size=2, colorscale='Viridis', 
                      line=dict(color='rgb(50,50,50)', width=0.25)))
    axis=dict(showbackground=False, showline=False, zeroline=False, showgrid=False, showticklabels=True)
    layout = go.Layout(
         autosize=True,
         #width=1000,height=1000,
         title=Myclasses[data.y],
         showlegend=False,
         scene=dict(aspectmode='data',xaxis=dict(axis),yaxis=dict(axis),zaxis=dict(axis))
        )        
    print("\nNodes:", numNodes)
    fig = go.Figure(data=[trace2], layout=layout)
    iplot(fig)

