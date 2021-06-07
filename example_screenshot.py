import open3d as o3d
import numpy as np
import os
from RGBDFace import *
from RGBDFace.utils import createPath

'''
在融合过程中逐帧截图

程序执行完成后，截图在数据集目录下的 screenshot 目录
'''


def getAniCallback(vis,gen,savePath):
    rgbdC=next(gen)
    lastGeo=rgbdC.tsdfC.tsdfMesh
    lastGeo.triangle_normals=o3d.utility.Vector3dVector()
    lastGeo.vertex_normals=o3d.utility.Vector3dVector()
    vis.add_geometry(lastGeo)
    createPath(savePath)
    
    def aniCallback(vis):
        nonlocal gen,lastGeo,rgbdC
        vis.capture_screen_image(os.path.join(savePath,rgbdC.colorFile)  )
        try:
            rgbdC=next(gen)
        except StopIteration:
            vis.register_animation_callback(None)
            return
        container=rgbdC.tsdfC
        mesh=container.tsdfMesh
        mesh.triangle_normals=o3d.utility.Vector3dVector()
        mesh.vertex_normals=o3d.utility.Vector3dVector()
        if lastGeo:
            vis.remove_geometry(lastGeo,reset_bounding_box=False)
        vis.add_geometry(mesh,reset_bounding_box=False)
        lastGeo=mesh
        vis.poll_events()
        return True
    return aniCallback

container= TSDFContainer(dataset="test1",keyframeInterval=50,minDepth=0,maxMeanDiff=200,sdfTrunc=0.005,registerMethods=[Methods.Ransac,Methods.RGBDOdometry])
gen=container.fullProcess3ByStep() # gen是个生成器

vis=o3d.visualization.Visualizer()
vis.create_window(width=1920,height=1080)
# opt = vis.get_render_option()
# opt.background_color = np.asarray([0, 1.0, 0])
# 绿背景
vis.register_animation_callback(getAniCallback(vis,gen,container.getFilePath("","screenshot")))
ctr=vis.get_view_control()
ctr.set_zoom(1.0)
ctr.set_front(np.asarray( [0.0,0.0,-1.0] ))
ctr.set_lookat(np.asarray( [0.0,0.0,0.3] ))
ctr.set_up(np.asarray( [0.0,-1.0,0.0] ))
try:
    vis.run()
    vis.close()
finally:
    vis.destroy_window()