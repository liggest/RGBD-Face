import open3d as o3d


FilterScope=o3d.geometry.FilterScope

def removePCDOutlier(pcd,voxel_size=0.001,nb_points=32,radius=0.004):
    '''
        尝试除去点云离群点
    '''
    downpcd=pcd.voxel_down_sample(voxel_size=voxel_size)
    inlierPcd,idxs=downpcd.remove_radius_outlier(nb_points=nb_points,radius=radius)
    return inlierPcd,idxs

def smoothMeshSimple(mesh,iterTimes=1):
    return mesh.filter_smooth_simple(number_of_iterations=iterTimes,filter_scope=FilterScope.Vertex)

def smoothMeshLaplacian(mesh,iterTimes=10,nLambda=0.85):
    return mesh.filter_smooth_laplacian(
        number_of_iterations=iterTimes,filter_scope=FilterScope.Vertex,
        **{"lambda":nLambda}
    )

def smoothMeshTaubin(mesh,iterTimes=30,nLambda=0.85,nMu=-0.25):
    return mesh.filter_smooth_taubin(number_of_iterations=iterTimes,filter_scope=FilterScope.Vertex,
        **{"lambda":nLambda,"mu":nMu}
    )

def postProcessMesh(mesh,smoothFunc,*args,**kw):
    mesh=mesh.remove_non_manifold_edges()
    mesh=mesh.remove_degenerate_triangles()
    mesh=mesh.remove_duplicated_triangles()
    mesh=mesh.remove_unreferenced_vertices()
    mesh=mesh.remove_duplicated_vertices()
    meshf=mesh.filter_sharpen(number_of_iterations=1,strength=0.05,filter_scope=FilterScope.Color)
    mesh.vertex_colors=meshf.vertex_colors
    mesh=smoothFunc(mesh,*args,**kw)
    mesh.compute_vertex_normals()
    return mesh
