import maya.cmds as cmds
import math
import maya.mel as mel

# delete unnessesary scene objects

groundList = cmds.ls('ground*')
boxList = cmds.ls('box*')
ParticleList = cmds.ls('particle*')
if len(groundList) > 0:
    cmds.delete(groundList)

if len(boxList) > 0:
    cmds.delete(boxList)
      
if len(ParticleList) > 0:
    cmds.delete(ParticleList)
    
    
#Added Lights

# Create a directional and ambient light
DirLight = cmds.directionalLight(rotation=(-45, 30, 15))
cmds.move(0,5,0,DirLight)
# Change the light intensity
cmds.directionalLight( DirLight, e=True, intensity=0.5 )

# Query it
cmds.directionalLight( DirLight, q=True, intensity=True )
# Result:0.5#

# Create an ambientLight light
AmbLight = cmds.ambientLight(intensity=0.3)
cmds.move(0,5,0,AmbLight)
# Change the light intensity
cmds.ambientLight( AmbLight, e=True, intensity=0.5 )

# Query it
cmds.ambientLight( AmbLight, q=True, intensity=True )
# Result:0.5 #


# ground and simulation box
ground = cmds.polyCube(w=10, h=1, d=10, name='ground#')
ground1 = cmds.polyCube(w=1, h=6, d=10, name='ground#')
ground2 = cmds.polyCube(w=10, h=6, d=1, name='ground#')
cmds.move(5, 1, 0, ground1)
cmds.move(0, 1, 5, ground2)

cube = cmds.polyCube(w=6 , h=3, d=4, name='box#')

cmds.move(0, 1, 0, cube)
cmds.select('box1.f[1]')
cmds.delete()

#Create induvidual materials for box and ground
def applyMaterial(node):
    if cmds.objExists(node):
        shd = cmds.shadingNode('lambert', name="%s_lambert" % node, asShader=True)
        shdSG = cmds.sets(name='%sSG' % shd, empty=True, renderable=True, noSurfaceShader=True)
        cmds.connectAttr('%s.outColor' % shd, '%s.surfaceShader' % shdSG)
        cmds.sets(node, e=True, forceElement=shdSG)    
applyMaterial("box1")
cmds.setAttr( "box1_lambert.transparency"  ,0.9,0.9,0.9,type = 'double3')

def applyMaterial(node):
    if cmds.objExists(node):
        shd = cmds.shadingNode('lambert', name="%s_lambert" % node, asShader=True)
        shdSG = cmds.sets(name='%sSG' % shd, empty=True, renderable=True, noSurfaceShader=True)
        cmds.connectAttr('%s.outColor' % shd, '%s.surfaceShader' % shdSG)
        cmds.sets(node, e=True, forceElement=shdSG)
        
applyMaterial("ground1")
cmds.setAttr( "ground1_lambert.color"  ,1,1,1,type = 'double3')

#Adding Spheres
count = 0
for i in range( 0, 35 ):
    for j in range( 0, 1 ):
        for k in range( 0, 23 ):
            count=count+1
            result = cmds.polySphere( r=0.08, sx=1, sy=1, name='particle#' )
            cmds.select('particle' + str(count)) 
            cmds.move(2.57-i*0.17+0.35, 1+j*0.17, 2.23-k*0.17-0.35,'particle' + str(count))

cmds.setAttr( 'lambert1.transparency', 0.7,0.7,0.7, type = 'double3' )
cmds.setAttr( 'lambert1.refractions', 1 )
cmds.setAttr( 'lambert1.refractiveIndex', 1.333 )
cmds.setAttr( 'lambert1.color', 0.56471 , 0.94118  ,0.86275, type = 'double3' )

#-----Fucntipns to main render loop-------#

#create poly6 smoothing kernel

def poly6Kernel(r, rj, h):
    # create resulting vector
    subtractedVec = []
    xVal = r[0] - rj[0]
    yVal = r[1] - rj[1]
    zVal = r[2] - rj[2]
    
    #length of vector
    length = math.sqrt(math.pow(xVal,2) + math.pow(yVal,2) + math.pow(zVal,2))
    
    #poly6kernel
    resPoly6Kernel = 315/(64*3.14*math.pow(h,9))*math.pow((math.pow(h,2)-math.pow(length,2)),3)
    
    if length >= 0 or length <= h:
        return resPoly6Kernel
    else:
        return 0.0
    
    
#res = poly6Kernel([3,2,2], [1,1,1], 2)
#print str(res)














