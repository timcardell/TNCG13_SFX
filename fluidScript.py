import maya.cmds as cmds
import sys

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

cmds.select('ground1')
cmds.setAttr('lambert3.transparency',0,0,0, type = 'double3' )

cube = cmds.polyCube(w=6 , h=2, d=4, name='box#')

cmds.move(0, 1, 0, cube)
cmds.select('box1.f[1]')
cmds.delete()

cmds.select('box1')
cmds.setAttr('lambert2.transparency',0.99, 0.99, 0.99, type = 'double3' )

#Adding Spheres
count = 0
for i in range( 0, 35 ):
    for j in range( 0, 1 ):
        for k in range( 0, 23 ):
            count=count+1
            result = cmds.polySphere( r=0.08, sx=1, sy=1, name='particle#' )
            cmds.select('particle' + str(count)) 
            cmds.move(2.57-i*0.17+0.35, 1+j*0.17, 2.23-k*0.17-0.35,'particle' + str(count))


