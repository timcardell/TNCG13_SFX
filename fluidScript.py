import maya.cmds as cmds
import math
import maya.mel as mel

# delete unnessesary scene objects

groundList = cmds.ls('ground*')
boxList = cmds.ls('box*')
ParticleList = cmds.ls('Particle*')
ListOfParticles = cmds.ls('Particle*',o=True, tr=True)
if len(groundList) > 0:
    cmds.delete(groundList)

if len(boxList) > 0:
    cmds.delete(boxList)

if len(ListOfParticles) > 0:
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
WidthParticles = 3#35
HeightParticles = 1
LenghtParticles = 3#23

for i in range( 0, WidthParticles ):
    for j in range( 0, HeightParticles ):
        for k in range( 0, LenghtParticles ):
            count=count+1
            result = cmds.polySphere( r=0.08, sx=1, sy=1, name='Particle#' )
            cmds.select('Particle' + str(count))
            cmds.move(2.57-i*0.17+0.35, 1+j*0.17, 2.23-k*0.17-0.35,'Particle' + str(count))

cmds.setAttr( 'lambert1.transparency', 0.7,0.7,0.7, type = 'double3' )
cmds.setAttr( 'lambert1.refractiveIndex', 1.333 )

cmds.setAttr( 'lambert1.color', 0.56471 , 0.94118  ,0.86275, type = 'double3' )








#-----Fucntipns to main simulation loop-------#


#create poly6 smoothing kernel


def lengthVec(x, y, z):
    return math.sqrt(math.pow(x,2) + math.pow(y,2) + math.pow(z,2))

def poly6Kernel(r, rj, h):
    # create resulting vector
    subtractedVec = []
    xVal = r[0] - rj[0]
    yVal = r[1] - rj[1]
    zVal = r[2] - rj[2]

    #length of vector

    length = lengthVec(xVal, yVal, zVal)
    #poly6kernel
    resPoly6Kernel = 315/(64*3.14*math.pow(h,9))*math.pow((math.pow(h,2)-math.pow(length,2)),3)

    if length >= 0 or length <= h:
        return resPoly6Kernel
    else:
        return 0.0



#res = poly6Kernel([3,2,2], [1,1,1], 2)
#print str(res)

def CalculateLambda(nrOfParticles, Pos, Neighbours, ZeroRho, EPSILON, h):
    
    Lambda = [0]* nrOfParticles
    
    for i in range (0, nrOfParticles):
        
        for j in range (1, len(Neighbours[i])):
           rjPos = Pos[Neighbours[j]]
           rho_i += poly6Kernel(Pos[i], rjPos, h)
        
    return Lambda


#Find neigboring particles within a radius rad

def findNeighboringParticles(nrOfParticles, Pos, rad):
    neighborMatrix = []
    neighborList = []
    epsilon = 0.0000000001
    
    for i in range (0,nrOfParticles):
       
        for j in range (0,nrOfParticles):
        
            particleDistance = lengthVec(Pos[i][0]-Pos[j][0], Pos[i][1]-Pos[j][1], Pos[i][2]-Pos[j][2])
            #print 'dist:  ' + str(particleDistance)
            
            if particleDistance > epsilon and particleDistance < rad:
                neighborList.append(j)
        
        neighborMatrix.append(neighborList)
    return neighborMatrix





#Simulation Loop

#Constants
dt = 0.0016
MaxSolverIterations = 40
rad = 0.3
ZeroRho = 1000

KeyFrames = 1
cmds.playbackOptions( playbackSpeed = 0, maxPlaybackSpeed = 1, min = 1, max = 150 )
startTime = cmds.playbackOptions( query = True, minTime = True )
endTime = cmds.playbackOptions( query = True, maxTime = True )
frame = startTime
numOfParticles = count






VelX = [0] * numOfParticles
VelY = [0] * numOfParticles
VelZ = [0] * numOfParticles

PredictedPosition = [0]*numOfParticles
Neighbours = [0]*numOfParticles
Lambda = [0]*numOfParticles
size = 0


for Particle in ListOfParticles:
    cmds.select(Particle)
    # only selecting the one sphere!
    pos = [ cmds.getAttr(".translateX"),
            cmds.getAttr(".translateY"),
            cmds.getAttr(".translateZ") ]
            
    PredictedPosition[size]= [pos[0],pos[1],pos[2]]

    size+=1

    cmds.setKeyframe(".translateX", value=pos[0], time=frame)
    cmds.setKeyframe(".translateY", value=pos[1], time=frame)
    cmds.setKeyframe(".translateZ", value=pos[2], time=frame)


  
for j in range (0,KeyFrames):    
    frame += 1
    #Predict position and velocities
    for i in range (0,numOfParticles):
         VelY[i]+= dt*9.82
         PredictedPosition[i][1] += dt*VelY[i]
                 
    #Find Neighboring particles     
    for l in range (0,numOfParticles):
          Neighbours[l] = findNeighboringParticles(numOfParticles,PredictedPosition, rad)
             
    Iter = 0
    while Iter < MaxSolverIterations : 
     
        for i in range (0,numOfParticles):
            Lambda[i] = CalculateLambda(numOfParticles, Pos, Neighbours, ZeroRho, EPSILON, h)
            
            
        
        
        
        Iter +=1
        
        
        
        

    
    
    
    
