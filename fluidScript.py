import maya.cmds as cmds

# delete unnessesary scene objects

groundList = cmds.ls('ground*')
boxList = cmds.ls('box*')

if len(groundList) > 0:
    cmds.delete(groundList)

if len(boxList) > 0:
    cmds.delete(boxList)


# ground and simulation box
ground = cmds.polyCube(w=10, h=1, d=10, name='ground#')

cube = cmds.polyCube(w=1, h=1, d=1, name='box#')

cmds.move(0, 1, 0, cube)


targetFace = cmds.select('box1.f[1]')
cmds.delete(targetFace)