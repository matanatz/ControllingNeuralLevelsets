import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.offline as offline
import torch
import numpy as np
from skimage import measure
import os
import trimesh
import utils.general as utils
import matplotlib.pyplot as plt

def get_threed_scatter_trace(points,caption = None,colorscale = None,color = None):

    assert points.shape[1] == 3, "3d scatter plot input points are not correctely shaped "
    assert len(points.shape) == 2, "3d scatter plot input points are not correctely shaped "

    trace = go.Scatter3d(
        x=points[:,0],
        y=points[:,1],
        z=points[:,2],
        mode='markers',
        name='projection',
        marker=dict(
            size=3,
            line=dict(
                width=2,
            ),
            opacity=0.9,
            colorscale=colorscale,
            showscale=True,
            color=color,
        ), text=caption)

    return trace

def plot_threed_scatter(points,path,epoch,in_epoch):
    trace = get_threed_scatter_trace(points)
    layout = go.Layout(width=1200, height=1200, scene=dict(xaxis=dict(range=[-2, 2], autorange=False),
                                                           yaxis=dict(range=[-2, 2], autorange=False),
                                                           zaxis=dict(range=[-2, 2], autorange=False),
                                                           aspectratio=dict(x=1, y=1, z=1)))

    fig1 = go.Figure(data=[trace], layout=layout)

    filename = '{0}/scatter_iteration_{1}_{2}.html'.format(path, epoch, in_epoch)
    offline.plot(fig1, filename=filename, auto_open=False)


def plot_manifold(points,decoder,path,epoch,in_epoch,resolution,mc_value,is_uniform_grid,verbose,save_html):
    if (is_uniform_grid):
        filename = '{0}/uniform_iteration_{1}_{2}.html'.format(path, epoch, in_epoch)
    else:
        filename = '{0}/nonuniform_iteration_{1}_{2}.html'.format(path, epoch, in_epoch)

    if (not os.path.exists(filename)):
        decoder.eval()
        pnts_val = decoder(points.cuda()).detach()
        caption = ["decoder : {0}".format(val.abs().mean().item()) for val in pnts_val.squeeze()]
        trace_pnts = get_threed_scatter_trace(points,caption=caption)
        trace_manifold = []
        surfaces = []

        for out in range(pnts_val.shape[-1]):
            surfaces.append(get_surface_trace(points,lambda x: decoder(x)[:,out].detach(),
                                              resolution,
                                              mc_value,
                                              is_uniform_grid,
                                              out,verbose))
            trace_manifold = trace_manifold + surfaces[-1]["mesh_trace"]
            surfaces[-1]['mesh_export'].export(filename.split('html')[0] + '_{0}.ply'.format(out), 'ply')


        layout = go.Layout(width=1200, height=1200, scene=dict(xaxis=dict(range=[-2, 2], autorange=False),
                                                               yaxis=dict(range=[-2, 2], autorange=False),
                                                               zaxis=dict(range=[-2, 2], autorange=False),
                                                               aspectratio=dict(x=1, y=1, z=1)))
        trace_curve = []
        if (pnts_val.shape[-1] == 2):
            curve = get_intersection(surfaces[0]['mesh_export'], lambda x: decoder(x)[:,1].detach())
            trace_curve = trace_curve + [curve["trace_curve"]]
            curve["curve"].export(filename.split('html')[0] + '_curve.ply', 'ply')



        fig1 = go.Figure(data=[trace_pnts] + trace_manifold + trace_curve, layout=layout)


        if (save_html):
            offline.plot(fig1, filename=filename, auto_open=False)

        decoder.train()


def get_surface_trace(points,decoder,resolution,mc_value,is_uniform,color_index,verbose):

    colors = ['#d8eff0', '#fff3b1']

    if (is_uniform):
        grid = get_grid_uniform(points,resolution)
    else:
        grid = get_grid(points,resolution)

    z = []

    for i,pnts in enumerate(torch.split(grid['grid_points'],100000,dim=0)):
        if (verbose):
            print ('{0}'.format(i/(grid['grid_points'].shape[0] // 100000) * 100))
        z.append(decoder(pnts).detach().cpu().numpy())
    z = np.concatenate(z,axis=0)

    if (not (np.min(z) > mc_value or np.max(z) < mc_value)):

        import trimesh
        z  = z.astype(np.float64)

        verts, faces, normals, values = measure.marching_cubes_lewiner(
            volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                             grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
            level=mc_value,
            spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1]))

        verts = verts + np.array([grid['xyz'][0][0],grid['xyz'][1][0],grid['xyz'][2][0]])
        meshexport = trimesh.Trimesh(verts, faces, normals, vertex_colors=values)

        def tri_indices(simplices):
            return ([triplet[c] for triplet in simplices] for c in range(3))

        I, J, K = tri_indices(faces)

        trace = [go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                          i=I, j=J, k=K, name='',
                          color=colors[color_index], opacity=0.5)]

    else:
        trace = []
        meshexport = None
    return {"mesh_trace":trace,
            "mesh_export":meshexport}

def plot_cuts(decoder,path,epoch,near_zero):
    onedim_cut = np.linspace(-1, 1, 1000)
    xx, yy = np.meshgrid(onedim_cut, onedim_cut)
    xx = xx.ravel()
    yy = yy.ravel()
    position_cut = np.vstack(([xx, np.zeros(xx.shape[0]), yy]))
    position_cut = [position_cut + np.array([0., i, 0.]).reshape(-1, 1) for i in np.linspace(-1.1, 0.7, 10)]
    for index, pos in enumerate(position_cut):
        #fig = tools.make_subplots(rows=1, cols=1)

        field_input = torch.tensor(pos.T, dtype=torch.float).cuda()
        z = []
        for i, pnts in enumerate(torch.split(field_input, 1000, dim=0)):
            input_ = pnts
            z.append(decoder(input_).detach().cpu().numpy())
        z = np.concatenate(z, axis=0)

        if (near_zero):
            trace1 = go.Contour(x=onedim_cut,
                                y=onedim_cut,
                                z=z.reshape(onedim_cut.shape[0], onedim_cut.shape[0]),
                                name='y = {0}'.format(pos[1, 0]),  # colorbar=dict(len=0.4, y=0.8),
                                autocontour=False,
                                contours=dict(
                                     start=-0.001,
                                     end=0.001,
                                     size=0.00001
                                     )
                                # ),colorbar = {'dtick': 0.05}
                                )
        else:
            trace1 = go.Contour(x=onedim_cut,
                                y=onedim_cut,
                                z=z.reshape(onedim_cut.shape[0], onedim_cut.shape[0]),
                                name='y = {0}'.format(pos[1, 0]),  # colorbar=dict(len=0.4, y=0.8),
                                autocontour=True,
                                # contours=dict(
                                #      start=-0.001,
                                #      end=0.001,
                                #      size=0.00001
                                #      )
                                # ),colorbar = {'dtick': 0.05}
                                )

        layout = go.Layout(width=1200, height=1200, scene=dict(xaxis=dict(range=[-1, 1], autorange=False),
                                                               yaxis=dict(range=[-1, 1], autorange=False),
                                                               aspectratio=dict(x=1, y=1)))
        # fig['layout']['xaxis2'].update(range=[-1, 1])
        # fig['layout']['yaxis2'].update(range=[-1, 1], scaleanchor="x2", scaleratio=1)

        filename = '{0}/cuts{1}_{2}.html'.format(path, epoch, index)
        fig1 = go.Figure(data=[trace1], layout=layout)
        offline.plot(fig1, filename=filename, auto_open=False)


def get_grid(points,resolution):
    eps = 0.01
    input_min = torch.min(points, dim=0)[0].squeeze().cpu().numpy()
    input_max = torch.max(points, dim=0)[0].squeeze().cpu().numpy()

    bounding_box = input_max - input_min
    shortest_axis = np.argmin(bounding_box)
    if (shortest_axis == 0):
        x = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(x) - np.min(x)
        y = np.arange(input_min[1] - eps, input_max[1] + eps, length / (x.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + eps, length / (x.shape[0] - 1))
    elif (shortest_axis == 1):
        y = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(y) - np.min(y)
        x = np.arange(input_min[0] - eps, input_max[0] + eps, length / (y.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + eps, length / (y.shape[0] - 1))
    elif (shortest_axis == 2):
        z = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(z) - np.min(z)
        x = np.arange(input_min[0] - eps, input_max[0] + eps, length / (z.shape[0] - 1))
        y = np.arange(input_min[1] - eps, input_max[1] + eps, length / (z.shape[0] - 1))

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float).cuda()
    return {"grid_points":grid_points,
            "shortest_axis_length":length,
            "xyz":[x,y,z],
            "shortest_axis_index":shortest_axis}

def get_grid_uniform(points,resolution):
    x = np.linspace(-1.2,1.2, resolution)
    y = x
    z = x

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float).cuda()

    return {"grid_points": grid_points,
            "shortest_axis_length": 2.4,
            "xyz": [x, y, z],
            "shortest_axis_index": 0}

def get_intersection(tri1, func2):
    verts1 = torch.tensor(tri1.vertices,  dtype=torch.float).cuda()
    faces1 = torch.tensor(tri1.faces, dtype = torch.long)

    v3 = func2(verts1)
    mask = v3 > 0
    outcount = torch.sum(mask[faces1], dim=1)
    cross = (outcount == 2) | (outcount == 1)
    crossing_tris = faces1[cross]
    out_vert = mask[crossing_tris]
    flip = out_vert.sum(dim=1) == 1
    out_vert[flip] = 1 - out_vert[flip]
    ntri = out_vert.shape[0]
    overt = torch.zeros(ntri, 3, dtype = torch.long)
    for i in range(ntri):
        v1i = torch.nonzero(~out_vert[i])[0,0]
        v2i = (1 + v1i) % 3
        v3i = (2 + v1i) % 3
        overt[i] = crossing_tris[i, [v1i,v2i,v3i]]

    u = -v3[overt[:, 0]] / (v3[overt[:, 1]] - v3[overt[:, 0]])
    v = -v3[overt[:, 0]] / (v3[overt[:, 2]] - v3[overt[:, 0]])
    uverts = (1 - u).unsqueeze(1).repeat([1,3]) * verts1[overt[:, 0],:] + u.unsqueeze(1).repeat([1,3]) * verts1[overt[:, 1],:]
    vverts = (1 - v).unsqueeze(1).repeat([1,3]) * verts1[overt[:, 0],:] + v.unsqueeze(1).repeat([1,3]) * verts1[overt[:, 2],:]

    x1 = np.ones((3, ntri))
    x1[0] = uverts[:, 0].cpu().numpy()
    x1[1] = vverts[:, 0].cpu().numpy()
    x1[2] = None

    y1 = np.ones((3, ntri))
    y1[0] = uverts[:, 1].cpu().numpy()
    y1[1] = vverts[:, 1].cpu().numpy()
    y1[2] = None

    z1 = np.ones((3, ntri))
    z1[0] = uverts[:, 2].cpu().numpy()
    z1[1] = vverts[:, 2].cpu().numpy()
    z1[2] = None

    verts_all = np.concatenate([uverts.cpu(), vverts.cpu()], axis=0)
    edges_all = np.vstack([np.arange(0,ntri), np.arange(ntri,ntri * 2), np.arange(0,ntri)]).T

    curve = trimesh.Trimesh(verts_all, edges_all)

    trace_curve = go.Scatter3d(
        name='both',
        x=np.ravel(x1, order='F'),
        y=np.ravel(y1, order='F'),
        z=np.ravel(z1, order='F'),
        mode='lines',
        line=dict(
            color='black',
            width=5
        )
    )

    return {"curve":curve, "trace_curve": trace_curve}

def plot_advimges(input,adv_samples,target,loss_fn,clone_model,plots_dir):
     with torch.no_grad():
        t = utils.get_cuda_ifavailable(
            torch.tensor(
                np.linspace(0,torch.norm(input - adv_samples,p=np.inf,dim=[1,2,3]).cpu().numpy(),100).T))

        pnts = input.unsqueeze(1) + t.reshape(t.shape[0], t.shape[1], 1, 1, 1) * (
                    adv_samples.unsqueeze(1) - input.unsqueeze(1)) / torch.norm(input - adv_samples, p=np.inf,
                                                                                dim=[1, 2, 3], keepdim=True).unsqueeze(1)

        clone_model.cpu()
        val_pnts = loss_fn(clone_model(pnts.reshape([-1] + list(pnts.shape[2:])).cpu()),
                                              target=target.unsqueeze(1).repeat(1,100).reshape(-1)).reshape(-1,100)

        num = 10
        fig = make_subplots(rows=num//2, cols=2)#, subplot_titles=np.arange(num).reshape(-1,2).tolist())



        def gallery(array, ncols=3):
            nindex, height, width, intensity = array.shape
            nrows = nindex // ncols
            assert nindex == nrows * ncols
            # want result.shape = (height*nrows, width*ncols, intensity)
            result = (array.reshape(nrows, ncols, height, width, intensity)
                      .swapaxes(1, 2)
                      .reshape(height * nrows, width * ncols, intensity))
            return result

        for i,x,y,img in zip(range(num),t[:num],val_pnts[:num],pnts[:num]):
            trace = go.Scatter(x=x.cpu().numpy(), y=y.cpu().numpy(), mode='lines', name='pgd_{0}'.format(i))

            # fig_images = make_subplots(10,10)
            # for index,curr_img in enumerate(img):
            #     trace_img = go.Scatter(
            #                         x=[0, img_width * scale_factor],
            #                         y=[0, img_height * scale_factor],
            #                         mode="markers",
            #                         marker_opacity=0)
            #     fig_images.add_trace(trace_img,index // 10 + 1, index % 10 + 1)
            #     fig_images.update_xaxes(visible=False, range=[0, img_width * scale_factor],row=index//10 + 1,col=index%10 + 1)
            #     fig_images.update_yaxes(visible=False, range=[0, img_height * scale_factor], scaleanchor="x",row=index//10  +1,col=index%10 + 1)


            fig.add_trace(trace,i//2 +1,i%2 + 1)
            gal = gallery(img[np.linspace(0, 99, 10)].cpu().numpy().transpose([0, 2, 3, 1]), 5)
            plt.imshow(gal.squeeze(), cmap='gray', vmin=0, vmax=1)
            plt.savefig(os.path.join(plots_dir,'epoch_{0}_data_{1}.png'.format(0,0)))

        fig.update_layout(
                autosize=False,
                width=1500,
                height=1200)
        offline.plot(fig,filename=os.path.join(plots_dir,'epoch_{0}_data_{1}.html'.format(0,0)),auto_open=False)
