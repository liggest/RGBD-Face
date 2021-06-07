from RGBDFace import *
import open3d as o3d
'''
特殊重建流程
允许不停地手动输入新RGBD图像
随时查看临时重建结果
此流程得到的重建结果可能相对粗糙一些，但过程更加灵活

推荐在jupyter notebook中执行
'''

container=TSDFContainer(dataset="test1",sdfTrunc=0.02,voxelSize=0.01,maxMeanDiff=200,cropEyeMouth=False)
container.addImgPair("IMG_5500.JPG","IMG_5499.PNG")
container.addImgPair("IMG_5502.JPG","IMG_5501.PNG")
container.addImgPair("IMG_5504.JPG","IMG_5503.PNG")
container.addImgPair("IMG_5506.JPG","IMG_5505.PNG")
container.addImgPair("IMG_5508.JPG","IMG_5507.PNG")
container.addImgPair("IMG_5510.JPG","IMG_5509.PNG")
container.addImgPair("IMG_5530.JPG","IMG_5529.PNG")
container.addImgPair("IMG_5532.JPG","IMG_5531.PNG")
container.addImgPair("IMG_5534.JPG","IMG_5533.PNG")
container.addImgPair("IMG_5536.JPG","IMG_5535.PNG")
container.addImgPair("IMG_5538.JPG","IMG_5537.PNG")
container.addImgPair("IMG_5540.JPG","IMG_5539.PNG")
container.addImgPair("IMG_5550.JPG","IMG_5549.PNG")
container.addImgPair("IMG_5552.JPG","IMG_5551.PNG")
container.addImgPair("IMG_5554.JPG","IMG_5553.PNG")
container.addImgPair("IMG_5556.JPG","IMG_5555.PNG")
container.addImgPair("IMG_5558.JPG","IMG_5557.PNG")
container.addImgPair("IMG_5560.JPG","IMG_5559.PNG")

o3d.visualization.draw_geometries([container.tsdfPCD])

container.save(description="addImgPair",sdfTrunc="sdfTrunc",voxelSize="voxelSize",meanD="maxMeanDiff")