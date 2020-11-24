import maya.cmds as cmds
import math
import maya.mel as mel

# delete unnessesary scene objects

groundList = cmds.ls('ground*')
boxList = cmds.ls('box*')
ListOfParticles = cmds.ls('Particle*', o=True, tr=True)
if len(groundList) > 0:
    cmds.delete(groundList)

if len(boxList) > 0:
    cmds.delete(boxList)

if len(ListOfParticles) > 0:
    cmds.delete(ListOfParticles)
    
ListOfParticles.insert(0,[])

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
epsilon = 200.0
correctionK = 0.001
correctionN = 4.0
correctionDeltaQ = 0.3
count = 0
WidthParticles = 5#35
HeightParticles = 5
LenghtParticles = 5#23
particleRadius = 0.08
for i in range( 0, WidthParticles ):
    for j in range( 0, HeightParticles ):
        for k in range( 0, LenghtParticles ): 
            count=count+1
            result = cmds.polySphere( r=particleRadius, sx=1, sy=1, name='Particle#' )
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
        
def poly6KernelAlternative(dQ, h):
    # create resulting vector


    #length = lengthVec(dQ[0], dQ[1], dQ[2])
    #poly6kernel
    resPoly6Kernel = 315/(64*3.14*math.pow(h,9))*math.pow((math.pow(h,2)-math.pow(dQ,2)),3)

    if dQ >= 0 or dQ <= h:
        return resPoly6Kernel
    else:
        return 0.0



#res = poly6Kernel([3,2,2], [1,1,1], 2)
#print str(res)


def spikyGrad(ri, rj, h):
    xVal = ri[0]-rj[0]
    yVal = ri[1]-rj[1]
    zVal = ri[2]-rj[2]

    
    r = [xVal, yVal, zVal]
    r_len= getLengthOfVec(r)
    r = norm(r)
    gradConstant = -45.0 / (math.pi * math.pow(h, 6))
    #print r
    if r >= 0 or r <= h:
    
        xGradient = gradConstant*math.pow(h-r_len, 2) * r[0]
        yGradient = gradConstant*math.pow(h-r_len, 2) * r[1]
        zGradient = gradConstant*math.pow(h-r_len, 2) * r[2]
        gradVec= [xGradient, yGradient, zGradient]

        return gradVec

    else:
        return [0.0, 0.0, 0.0]


def vecMult(Vec1, Vec2):
    tempVec = Vec1[0]*Vec2[0]+Vec1[1]*Vec2[1]+Vec1[2]*Vec2[2]
    return tempVec

def scalarMult(Vec1, scalar):
    tempVec = [Vec1[0]*scalar,Vec1[1]*scalar,Vec1[2]*scalar]
    return tempVec

def addVect(Vec1, Vec2):
    tempVec = [Vec1[0]+Vec2[0],Vec1[1]+Vec2[1],Vec1[2]+Vec2[2]]
    return tempVec

def subVect(Vec1, Vec2):
    tempVec = [Vec1[0]-Vec2[0],Vec1[1]-Vec2[1],Vec1[2]-Vec2[2]]
    return tempVec

def addScalarVect(Vec1, scalar):
    tempVec = [Vec1[0]+scalar,Vec1[1]+scalar,Vec1[2]+scalar]
    return tempVec

def projectVect(Vec1, Vec2) :
    c_1 = vecMult(Vec1, Vec2)
    c_2 = vecMult(Vec2, Vec2)

    tempVec = scalarMult(Vec1, c_1)
    scalarMult(tempVec, 1.0/c_2)

    return tempVec





# create lambda from equation ......

def CalculateLambda(nrOfParticles, predictedPositions, Neighbours, ZeroRho, EPSILON, h):
    rho_i = 0
    C_i = 0
    sumGradient = [0,0,0]
    Lambda = [0]* nrOfParticles

    for i in range (0, nrOfParticles):
        Pos = predictedPositions[i]
        for j in range (0, len(Neighbours[i])):
           cmds.select(ListOfParticles[Neighbours[i][j]])
           rjPos = [cmds.getAttr(".translateX"),cmds.getAttr(".translateY"),cmds.getAttr(".translateZ") ]
           #print 'pos' +str(Pos)
           #print 'posj' + str(rjPos)
           
           rho_i += poly6Kernel(Pos, rjPos, h)
           
           gradientPk = spikyGrad(Pos, rjPos, h)

           kernel =  poly6Kernel(Pos, rjPos, h)

           kernelDone = scalarMult(gradientPk,kernel)
           
           sumGradient = addVect(sumGradient,kernelDone)

        sumGradient = scalarMult(sumGradient,(1/ZeroRho))

        dotSum = vecMult(sumGradient,sumGradient)


        C_i = (rho_i / ZeroRho)- 1

        Lambda[i] = ((-1)*C_i)/(dotSum + EPSILON)

    return Lambda

#calculate delta position eq 12 in m�ller paper

def norm(VecIn):
    length=getLengthOfVec(VecIn)
    return scalarMult( VecIn,1.0 / length) 

def getLengthOfVec(VecIn) :
    return math.sqrt(vecMult(VecIn,VecIn))


def deltaP(lambdaa, rho_0, numOfParticles, pos, h, neighbours):
    deltaPos = []
    if len(deltaPos) > 0:
        deltaPos[:] = []
    deltaPos.insert(0,[])       
    newCorr = 0
    corrPoss = 0
    for i in range (1,numOfParticles):
        lambdaI = lambdaa[i]
        posi = pos[i]
        sumX = 0
        sumY = 0
        sumZ = 0
        for j in range (1,len(neighbours[i])):
            cmds.select(ListOfParticles[neighbours[i][j]])
            posj = [cmds.getAttr(".translateX"),cmds.getAttr(".translateY"),cmds.getAttr(".translateZ") ]
           
            lambdaJ = lambdaa[neighbours[i][j]]
            lambdaIJ = lambdaJ + lambdaI
            newCorr = computeCorrectionScore(0.001, h, 4, 0.3, posi, posj)
            spikyGradient = spikyGrad(posi, posj, h)
            
            sumX = sumX +  (newCorr + lambdaIJ)*(spikyGradient[0])
            sumY = sumY +  (newCorr + lambdaIJ)*(spikyGradient[1]) 
            sumZ = sumZ +  (newCorr + lambdaIJ)*(spikyGradient[2])
                
            sumX  = sumX * (1/rho_0)
            sumY  = sumY * (1/rho_0)
            sumZ  = sumZ * (1/rho_0)
            corrPoss = [sumX, sumY, sumZ]
           

        deltaPos.append(corrPoss)
     
    return deltaPos

#Find neigboring particles within a radius rad
def computeCorrectionScore(k, h, n, dQ, p1, p2) :
    constraint = poly6Kernel(p1,p2,h) / poly6KernelAlternative(dQ*h,h)
    return -k * math.pow(constraint, n)


def findNeighboringParticles(nrOfParticles, Pos, rad):
    neighborMatrix = []
    
    epsilon = 0.001

    for i in range (1,nrOfParticles):
        neighborList = []
        for j in range (1,nrOfParticles):
            particleDistance = lengthVec(Pos[i][0]-Pos[j][0], Pos[i][1]-Pos[j][1], Pos[i][2]-Pos[j][2])
           # print 'dist:  ' + str(particleDistance)

            if particleDistance > epsilon and particleDistance < rad:
                neighborList.append(j)
                
            
            #print str(neighborList)
        neighborMatrix.append(neighborList)
        
    neighborMatrix.insert(0,[]) 
    
    return neighborMatrix


def BoxConstraints(Pos,Vel,Rad,numOfParticles):

    xMin = -5+ Rad
    xMax = 5 - Rad
    yMin = 0.5
    yMax = 5
    zMin = -5 + Rad
    zMax = 5 - Rad

    for i in range (1,numOfParticles):
        if Pos[i][0] < xMin :
            Pos[i][0] = xMin
            Vel[i][0] = 0.0
        elif Pos[i][0] > xMax :
            Pos[i][0] = xMax
            Vel[i][0] = 0.0

        if Pos[i][1] < yMin :
            Pos[i][1] = yMin
            Vel[i][1] = 0.0
        elif Pos[i][1] > yMax :
            Pos[i][1] = yMax
            Vel[i][1] = 0.0
            
        if Pos[i][2] < zMin :
            Pos[i][2] = zMin
            Vel[i][2] = 0.0
        elif Pos[i][2] > zMax :
            Pos[i][2] = zMax
            Vel[i][2] = 0.0
    return [Pos, Vel]




def calculateParticleCollisionResponse(Pos, Vel, Rad, Neighbours,numOfParticles):
    
    for i in range (1, numOfParticles):
        posI = Pos[i]
        for j in range (1, len(Neighbours[i])):
           cmds.select(ListOfParticles[Neighbours[i][j]])
           posJ = [cmds.getAttr(".translateX"),cmds.getAttr(".translateY"),cmds.getAttr(".translateZ")]
           #print posI
           
           particleDistance = subVect(posI,posJ)
           particleDistance = lengthVec(particleDistance[0],particleDistance[1],particleDistance[2])

           if particleDistance > 0 and particleDistance <= 2.0 * Rad :
               velI = Vel[i]
               velJ = [Vel[Neighbours[i][j]][0],Vel[Neighbours[i][j]][1],Vel[Neighbours[i][j]][2]]

               dBetweenParticles = subVect(posI,posJ)
               
               collisionPoint = addVect(posI, dBetweenParticles)

               d = lengthVec(dBetweenParticles[0],dBetweenParticles[1],dBetweenParticles[2])

               dBetweenParticles = norm(dBetweenParticles)

               negativeVec = scalarMult(scalarMult(dBetweenParticles,-1),0.1)
             
               Pos[i] = addVect(Pos[i],negativeVec)

               
               Pos[Neighbours[i][j]] = addVect(Pos[Neighbours[i][j]],scalarMult(dBetweenParticles,0.1))

               newVels = calcVelocity(posI,posJ, velI, velJ)

               Vel[i][0] = newVels[0][0]
               Vel[i][1] = newVels[0][1]
               Vel[i][2] = newVels[0][2]

               Vel[Neighbours[i][j]][0] = newVels[1][0]
               Vel[Neighbours[i][j]][1] = newVels[1][1]
               Vel[Neighbours[i][j]][2] = newVels[1][2]

  
    return [Pos, Vel]

def calcVelocity(posI,posJ,velI,velJ):
    NewPosIJ = subVect(posI, posJ)
    NewPosJI = subVect(posJ, posI)

    newVel1 =  projectVect(velJ,NewPosJI)
    newVel2 =  projectVect(velI,NewPosIJ)

    newVel1 = subVect(newVel1, projectVect(velI, NewPosIJ))
    newVel2 = subVect(newVel2, projectVect(velJ, NewPosJI))

    return [newVel1, newVel2]

# calclulate vorticiity confinement
def vorticityConfinement(predictedVelocity, predictedPosition, neighbours, h, numOfParticles):

    vorticityVec=[]
    
    for i in range (1,numOfParticles):
        #calculate position and velocity for all particles
        posi = predictedPosition[i]
        veli = predictedVelocity[i]
        
        for j in range (1, len(neighbours[i])):
          #calculate velocity and position from i's neighbors
          velj = [predictedVelocity[neighbours[i][j]][0], predictedVelocity[neighbours[i][j]][1], predictedVelocity[neighbours[i][j]][2]]
          cmds.select(ListOfParticles[Neighbours[i][j]])
          posJ = [cmds.getAttr(".translateX"),cmds.getAttr(".translateY"),cmds.getAttr(".translateZ")]

          #calculate vij from eq 15 in m�ller
          
          vij = [velj[0]-veli[0], velj[1]-veli[1], velj[2]-veli[2]]
          
          #print 'velj'+str(veli)
          #spiky gradient, inserted in eq 15
          grad = spikyGrad(posi, posJ, h)

          #cross poduct of gradient and vij
          crossProduct = [(vij[1]*grad[2])-vij[2]*grad[1], -(vij[0]*grad[2])-vij[2]*grad[0], (vij[0]*grad[1])-vij[1]*grad[0]]
          vorticityVec.append(crossProduct)
    #vorticityVec = crossProduct
    vorticityVec.insert(0,[])
    return vorticityVec

def applyXSPH(c, h, Pos, Vel, neighbours,numOfParticles):
    vNew = [0,0,0]
    sumJ = [0,0,0]
    for i in range (1, numOfParticles):
        posI = Pos[i]
        velI = Vel[i]
        for j in range (1, len(neighbours[i])):
              cmds.select(ListOfParticles[Neighbours[i][j]])
              posJ = [cmds.getAttr(".translateX"),cmds.getAttr(".translateY"),cmds.getAttr(".translateZ")]
              velJ = [Vel[Neighbours[i][j]][0],Vel[Neighbours[i][j]][1],Vel[Neighbours[i][j]][2]]
              
              W = poly6Kernel(posI,posJ, h)
              vIJ = subVect(velI,velJ)

              sumJ += scalarMult(vIJ,W)
              sumJ = scalarMult(sumJ,c)
              
              vNew = addVect(velI,sumJ)
              
              
   
    return vNew
    

#compute corrective force eq 16
def fVorticity(vorticity, particlePosition, epsilon, h,Neighbours):
    fVorticity = []
    
    for i in range (1,numOfParticles):
        posi = particlePosition[i]
        
        for j in range (1, len(Neighbours[i])):
        
          vort = vorticity[Neighbours[i][j]]
          
          vortLen = lengthVec(vort[0],vort[1],vort[2])
          cmds.select(ListOfParticles[Neighbours[i][j]])
          posJ = [cmds.getAttr(".translateX"),cmds.getAttr(".translateY"),cmds.getAttr(".translateZ")]
          grad = spikyGrad(posi, posJ, h)
          
          n = scalarMult(grad,vortLen)
          nNormFactor = lengthVec(n[0],n[1],n[2])
          #print nNormFactor
          if nNormFactor < 0.00000001:
              nNormFactor = 0.000001
          N = scalarMult(n,(1/nNormFactor))
          
          crossProduct = [(N[1]*vort[2])-N[2]*vort[1], -(N[0]*vort[2])-N[2]*vort[0], (N[0]*vort[1])-N[1]*vort[0]]
          fVorticity.append(crossProduct)
    fVorticity.insert(0,[])
    return fVorticity


#Simulation Loop

#Constants
dt = 0.0016
MaxSolverIterations = 3
rad = 0.3
ZeroRho = 200
h = 0.4
c = 0.001
EPSILON = 200
KeyFrames = 50
cmds.playbackOptions( playbackSpeed = 0, maxPlaybackSpeed = 1, min = 1, max = 150 )
startTime = cmds.playbackOptions( query = True, minTime = True )
endTime = cmds.playbackOptions( query = True, maxTime = True )
frame = startTime
numOfParticles = count+1

particleVelocity = [0]
ppx = [0] *numOfParticles
ppy = [0] *numOfParticles
ppz = [0] *numOfParticles

PredictedPosition = []
Neighbours = []
Lambda = []
size = 0

defaultVel = 0

if len(PredictedPosition) > 0:
    PredictedPosition[:] = []

if len(Neighbours) > 0:
    Neighbours[:] = []

if len(Lambda) > 0:
    Lambda[:] = []

if len(particleVelocity) > 0:
    particleVelocity[:] = []

particleVelocity.insert(0,[])
Neighbours.insert(0,[])
Lambda.insert(0,[])
PredictedPosition.insert(0,[])


for i in range ( 1, numOfParticles ):
    cmds.select(ListOfParticles[i])
    
    pos = [cmds.getAttr(".translateX"), cmds.getAttr(".translateY"),cmds.getAttr(".translateZ")]
   
    ppx[i] = pos[0]
    ppy[i] = pos[1]
    ppz[i] = pos[2]
    
    cmds.setKeyframe(".translateX", value=pos[0], time=frame)
    cmds.setKeyframe(".translateY", value=pos[1], time=frame)
    cmds.setKeyframe(".translateZ", value=pos[2], time=frame)

for i in range ( 1, numOfParticles ):
    PredictedPosition.append([ppx[i],ppy[i],ppz[i]])
    particleVelocity.append([0,0,0])
   
    

for j in range (1,KeyFrames):
    print 'Frame: ' + str(frame)
    frame += 1

    #Predict position and velocities
    for i in range (1,numOfParticles):
        
             particleVelocity[i][1] = particleVelocity[i][1] - (dt*9.82)
             PredictedPosition[i][1] += dt*particleVelocity[i][1]
        
    #Create Bounding box and bounding conditions
    
    Constraints = []
    Constraints = BoxConstraints(PredictedPosition,particleVelocity,particleRadius,numOfParticles)
        
    for i in range (1,numOfParticles):
             PredictedPosition[i][0] = Constraints[0][i][0]
             PredictedPosition[i][1] = Constraints[0][i][1]
             PredictedPosition[i][2] = Constraints[0][i][2]
             particleVelocity[i][0] = Constraints[1][i][0]
             particleVelocity[i][1] = Constraints[1][i][1]
             particleVelocity[i][2] = Constraints[1][i][2]
    
        #Find Neighboring particles
        
    Neighbours = findNeighboringParticles(numOfParticles,PredictedPosition, rad)

    Iter = 0
    while Iter < MaxSolverIterations :
            
        Lambda = CalculateLambda(numOfParticles, PredictedPosition, Neighbours, ZeroRho, EPSILON, h)
            
        deltaPositions = deltaP(Lambda, ZeroRho, numOfParticles, PredictedPosition, h,Neighbours)
       
        for i in range (1 , numOfParticles):
            PredictedPosition[i] =addVect(PredictedPosition[i],deltaPositions[i])
      
        particleCollision = calculateParticleCollisionResponse(PredictedPosition, particleVelocity, particleRadius, Neighbours,numOfParticles)
        
        for i in range (1 , numOfParticles):
            
            PredictedPosition[i][0] = particleCollision[0][i][0]
            PredictedPosition[i][1] = particleCollision[0][i][1]
            PredictedPosition[i][2] = particleCollision[0][i][2]
            particleVelocity[i][0] = particleCollision[1][i][0]
            particleVelocity[i][1] = particleCollision[1][i][1]
            particleVelocity[i][2] = particleCollision[1][i][2]
        
       # Constraints = BoxConstraints(PredictedPosition,particleVelocity,particleRadius,numOfParticles)
        
        for i in range (1 , numOfParticles):
             
             PredictedPosition[i][0] = Constraints[0][i][0]
             PredictedPosition[i][1] = Constraints[0][i][1]
             PredictedPosition[i][2] = Constraints[0][i][2]
             particleVelocity[i][0] = Constraints[1][i][0]
             particleVelocity[i][1] = Constraints[1][i][1]
             particleVelocity[i][2] = Constraints[1][i][2]
        
        Iter +=1
    #End while Loop    
    #print  particleVelocity       
    
    for n in range (1,numOfParticles):
        cmds.select(ListOfParticles[n])
        pos = [cmds.getAttr(".translateX"), cmds.getAttr(".translateY"),cmds.getAttr(".translateZ")]
        particleVelocity[n] = scalarMult(subVect(PredictedPosition[n], pos), (1/dt))
    
    vort = vorticityConfinement(particleVelocity, PredictedPosition, Neighbours, h, numOfParticles)
    f_Vorticity = fVorticity(vort, PredictedPosition, EPSILON, h,Neighbours)
    XSPH = applyXSPH(c, h, PredictedPosition, particleVelocity, Neighbours,numOfParticles)
    
    for i in range (1, numOfParticles):
         particleVelocity[i] =  addVect(particleVelocity[i],scalarMult(f_Vorticity[i],dt))
         
       
    for i in range (1, numOfParticles) :
        cmds.select( 'Particle'+str(i) )
        cmds.setKeyframe(".translateX", value=PredictedPosition[i][0], time=frame)
        cmds.setKeyframe(".translateY", value=PredictedPosition[i][1], time=frame)
        cmds.setKeyframe(".translateZ", value=PredictedPosition[i][2], time=frame)


        
        
        
        
