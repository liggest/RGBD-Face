import os
import sys
import functools

def checkPath(path,raiseErr=True):
    '''
        检查路径是否存在，不存在时则由raiseErr决定是否抛出错误
    '''
    if not os.path.exists(path):
        if raiseErr:
            raise FileExistsError(f"没有找到 {path}")
        else:
            print(f"【Error】没有找到 {path}",file=sys.stderr)


def createPath(path):
    '''
        若目录不存在则创建，保证目录存在
    '''
    if not os.path.exists(path):
        os.mkdir(path)
        print("创建了路径",path)
    return path

def expandFileName(fileName,prefix="",suffix="",ext=None):
    '''
        为文件名添加前缀、后缀文本，还可以改变扩展名
    '''
    path= [ *os.path.split(fileName) ]
    name,oldExt=os.path.splitext(path[-1])
    if ext is None:
        ext=oldExt
    elif ext and isinstance(ext, str) and not ext.startswith("."): #如果ext为空则默认无后缀
        ext=f".{ext}"
    name=f"{prefix}{name}{suffix}{ext}"
    path[-1]=name
    return os.path.join(*path)

def deprecated(msg="已被废弃"):
    '''
        函数废弃提示
    '''
    def warpper(func):
        @functools.wraps(func)
        def inner(*args,**kw):
            name=getattr(func,"__name__") or repr(func)
            print(f"{name} {msg}")
            return func(*args,**kw)
        return inner
    return warpper