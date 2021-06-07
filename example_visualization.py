import open3d as o3d
import numpy as np

'''
    使用Open3D看模型
    还能让模型转一转
'''

def easeIn(x):
    ''' 平滑移动 '''
    return 2*x**2 if x<0.5 else 1-(-2*x+2)**2 /2

def rotateAnimation(mesh):
    step=0
    maxStep=60
    lastDeg=0
    amp=180
    
    def rotateView(vis):
        ''' 左右转 '''
        nonlocal step,lastDeg,mesh,amp
        x=step/maxStep
        ctr = vis.get_view_control()
        deg=amp*easeIn(x)
        ctr.rotate(deg-lastDeg, 0)
        lastDeg=deg
        step+=1
        if step>maxStep:
            step=0
            lastDeg=0
            if amp==180:
                amp*=2
            amp*=-1
        vis.poll_events()
        return True
    
    def rotateView2(vis):
        ''' 转圈 '''
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 1.0, 0])
        ctr = vis.get_view_control()
        ctr.rotate(10.0, 0.0)
        return False
    
    def startAnimation(vis):
        vis.register_animation_callback(rotateView)
        return False
    
    #o3d.visualization.draw_geometries_with_animation_callback([mesh],rotateView)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 1.0, 0]) # 绿背景
    vis.add_geometry(mesh)
    vis.register_key_callback(ord('K'),startAnimation) # 按K开始转
    ctr=vis.get_view_control()
    ctr.set_zoom(1.0)
    ctr.set_front(np.asarray( [0.0,0.0,-1.0] ))
    ctr.set_lookat(np.asarray( [0.0,0.0,0.3] ))
    ctr.set_up(np.asarray( [0.0,-1.0,0.0] ))
    #vis.register_animation_callback(rotateView)
    try:
        vis.run()
        vis.close()
    finally:
        vis.destroy_window()
        print("关闭了！")

# 读取网格（示例）
mesh=o3d.io.read_triangle_mesh("./datasets/test1/result/test1 fullProcess3_Odo key=50 mind=0 meanD=200 sdfTrunc=0.005 mesh smoothed.ply")

# 除去法线
mesh.triangle_normals=o3d.utility.Vector3dVector()
mesh.vertex_normals=o3d.utility.Vector3dVector()

# 按K开始转
rotateAnimation(mesh)