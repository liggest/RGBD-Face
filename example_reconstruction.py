from RGBDFace import *

'''
设置数据集根目录：
TSDFContainer.default["datasetBase"]="./datasets/"

基本步骤：
container=TSDFContainer(dataset="xxx") 指定数据集、各种参数，创建对象
    参数设置详见 TSDFContainer.default
container.fullProcess3() 执行重建流程
    可传入名为 fileSlice 的 slice 对象，对图像序列切片，只是用切得的部分
    默认 fileSlice=slice(None,None,None)
container.save(description="文件名中的一些描述") 保存重建结果
    保存时可指定文件名中附带一些参数信息
    在 save 方法中传入：
    参数在程序中的名称 = True  或者  参数在文件名中的名称 = "参数在程序中的名称"

中间文件在数据集目录下的 crop 目录
    若参数改动较大导致需要重新生成中间文件，执行重建前删除该目录即可
重建结果在数据集目录下的 result 目录
    还可在 meshlab 中进一步做网格平滑
'''

container=TSDFContainer(dataset="test1",keyframeInterval=25,sdfTrunc=0.02,voxelSize=0.01,registerMethods=[Methods.Ransac,Methods.RGBDOdometry],maxMeanDiff=200)
container.fullProcess3()
container.save(description="fullProcess3_Odo",key="keyframeInterval",sdfTrunc="sdfTrunc",voxelSize="voxelSize",meanD="maxMeanDiff")