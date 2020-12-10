import maya.cmds as cmds
import math
import maya.mel as mel
# delete unnessesary scene objects


ListOfParticles = cmds.ls('Particle*', o=True, tr=True)

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
#Adding Spheres

count = 0
WidthParticles = 5#35
HeightParticles = 20
LenghtParticles = 5#23
particleRadius = 0.05
for i in range( 0, WidthParticles ):
    for j in range( 0, HeightParticles ):
        for k in range( 0, LenghtParticles ): 
            count=count+1
            result = cmds.polySphere( r=particleRadius, sx=1, sy=1, name='Particle#' )
            cmds.select('Particle' + str(count)) 
            cmds.move(-i*0.11, 0.4+j*0.11, k*0.11,'Particle' + str(count))
            

# Create transparent material for transparent boxs
cmds.setAttr( 'lambert1.refractions', 1 )
cmds.setAttr( 'lambert1.refractiveIndex', 1.33 )
cmds.setAttr( 'lambert1.color', 0.1, .1, 1, type = 'double3' )


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
   
    r_len= lengthVec(r[0],r[1],r[2])
    
    if r_len <= 0.0 or r_len >= h :
        return [0.0, 0.0, 0.0]
    
    r = norm(r)
    gradConstant = -45.0 / (math.pi * math.pow(h, 6))
    
  
    
    xGradient = gradConstant*math.pow(h-r_len, 2) * r[0]
    yGradient = gradConstant*math.pow(h-r_len, 2) * r[1]
    zGradient = gradConstant*math.pow(h-r_len, 2) * r[2]
    gradVec= [xGradient, yGradient, zGradient]

    return gradVec


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

def CalculateLambda(nrOfParticles, predictedPositions, Neighbours, ZeroRho, h):
    C_i = 0
    sumGradient = [0,0,0]
    Lambda = []
    dotSum = 0
    
    for i in range (1, nrOfParticles):
        Pos = predictedPositions[i]
        rho_i = 0
                    
        for j in range (1, len(Neighbours[i])):
           rjPos = [predictedPositions[Neighbours[i][j]][0], predictedPositions[Neighbours[i][j]][1], predictedPositions[Neighbours[i][j]][2]]

           rho_i += poly6Kernel(Pos, rjPos, h)
           C_i = (rho_i / ZeroRho) - 1
            
           gradientPk = spikyGrad(Pos, rjPos, h)

           kernel =  poly6Kernel(Pos, rjPos, h)

           kernelDone = scalarMult(gradientPk,kernel)
           
           sumGradient = addVect(sumGradient,kernelDone)

           sumGradient = scalarMult(sumGradient,(-1/ZeroRho))
            
           dotSum += vecMult(sumGradient,sumGradient)
        res = ((-1)*C_i)/(dotSum + 0.1)      

        Lambda.append(res)
    Lambda.insert(0,[])    

    return Lambda

#calculate delta position eq 12 in m�ller paper

def norm(VecIn):
    length=lengthVec(VecIn[0],VecIn[1],VecIn[2])
    return scalarMult( VecIn,1.0 / length) 

def getLengthOfVec(VecIn) :
    return math.sqrt(vecMult(VecIn,VecIn))

def deltaP(lambdaa, rho_0, numOfParticles, pos, h, neighbours):
    deltaPos = []
    newCorr = 0
    corrPoss = 0
    alpha = 2
    for i in range (1,numOfParticles):
        lambdaI = lambdaa[i]
        posi = pos[i]
        sumX = 0
        sumY = 0
        sumZ = 0
        for j in range (1,len(neighbours[i])):
            posj = [pos[neighbours[i][j]][0], pos[neighbours[i][j]][1], pos[neighbours[i][j]][2]]
           
            newCorr = computeCorrectionScore(0.1, h, 4, 0.1, posi, posj)
            
            spikyGradient = spikyGrad(posi, posj, h)
            
            sumX = sumX + ( lambdaI * (spikyGradient[0]))
            sumY = sumY + ( lambdaI * (spikyGradient[1]))
            sumZ = sumZ + ( lambdaI * (spikyGradient[2]))
                
            sumX  = sumX * -alpha * (1/rho_0)
            sumY  = sumY * -alpha * (1/rho_0)
            sumZ  = sumZ * -alpha * (1/rho_0)
            corrPoss = [sumX, sumY, sumZ]
           

        deltaPos.append(corrPoss)
    deltaPos.insert(0,[])      
    return deltaPos

#Find neigboring particles within a radius rad
def computeCorrectionScore(k, h, n, dQ, p1, p2) :
    constraint = poly6Kernel(p1,p2,h) / poly6KernelAlternative(dQ*h,h)
    return -k * math.pow(constraint, n)


def findNeighboringParticles(nrOfParticles, Pos, rad):
    neighbours = []
    neighbours.append([0]) # dummy

    for i in range ( 1, nrOfParticles ): 
        closestNeighbours = []

        for j in range (1, nrOfParticles ):
            if i == j:
                continue
            
            arrayDistance = subVect(Pos[i],Pos[j])
            distance = math.sqrt(math.pow(arrayDistance[0], 2) + math.pow(arrayDistance[1], 2) + math.pow(arrayDistance[2], 2))

            if(distance < rad) :
                closestNeighbours.append(j)
        
        neighbours.append(closestNeighbours)
        
    return neighbours



def BoxConstraints(Pos,Vel,Rad,numOfParticles,variable):

    xMin = -1+ Rad
    xMax = 0.7 - Rad
    yMin = 0
    yMax = 6
    zMin = -1.5 + Rad
    zMax = 0.7 - Rad

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
            Vel[i][1] = -9.82
            
        if Pos[i][2] < zMin :
            Pos[i][2] = zMin
            Vel[i][2] = 0.0
        elif Pos[i][2] > zMax :
            Pos[i][2] = zMax
            Vel[i][2] = 0.0
    return [Pos, Vel]




def calculateParticleCollisionResponse(Pos, Vel, Rad, Neighbours,numOfParticles):
    offset = 0.01

    for i in range (1, numOfParticles) :
        pos_i = Pos[i]
        v_i = [Vel[i][0], Vel[i][1], Vel[i][2]]
        
        for j in range (1, len(Neighbours[i])) :
            pos_j = [Pos[Neighbours[i][j]][0], Pos[Neighbours[i][j]][1], Pos[Neighbours[i][j]][2]]
            posIJ = subVect(pos_j,pos_i)
            distance =math.sqrt(vecMult(posIJ,posIJ))
           
            if distance <= (2 * Rad) and distance > 0.0: # COLLISION!
                v_j = [Vel[Neighbours[i][j]][0],  Vel[Neighbours[i][j]][1],  Vel[Neighbours[i][j]][2]]

                vecBetweenParticles = subVect(pos_j,pos_i)
                collisionPoint = addVect(pos_i, vecBetweenParticles)
                
                vecBetweenParticles = norm(vecBetweenParticles)

                negVec = scalarMult(scalarMult(vecBetweenParticles, -1.0), offset)
                
                Pos[i][0] = Pos[i][0] + negVec[0]
                Pos[i][1] = Pos[i][1] + negVec[1]
                Pos[i][2] = Pos[i][2] + negVec[2]
                
                Pos[Neighbours[i][j]][0] = Pos[Neighbours[i][j]][0] + offset * vecBetweenParticles[0]
                Pos[Neighbours[i][j]][1] = Pos[Neighbours[i][j]][1] + offset * vecBetweenParticles[1]
                Pos[Neighbours[i][j]][2] = Pos[Neighbours[i][j]][2] + offset * vecBetweenParticles[2]

                newVels = CalcVel(pos_i, pos_j, v_i, v_j)
           
                
                Vel[i][0] = newVels[0][0]
                Vel[i][1] = newVels[0][1]
                Vel[i][2] = newVels[0][2]
                
                Vel[Neighbours[i][j]][0] = newVels[1][0]
                Vel[Neighbours[i][j]][1] = newVels[1][1]
                Vel[Neighbours[i][j]][2] = newVels[1][2]
                

    return [Pos, Vel]
    
def CalcVel(PosI,PosJ,VelI,VelJ):

    n = subVect(PosI,PosJ)
    n = norm(n)

    a1 = vecMult(VelI,n);
    a2 = vecMult(VelJ,n);

    optimizedP = (2.0 * (a1 - a2))

    V1 = subVect(VelI,scalarMult( n, optimizedP))

    V2 = addVect(VelI,scalarMult( n, optimizedP))

    return [V1,V2]

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
          posJ =  [predictedPosition[Neighbours[i][j]][0], predictedPosition[Neighbours[i][j]][1], predictedPosition[Neighbours[i][j]][2]]

          #calculate vij from eq 15 in muller
          
          vij = [velj[0]-veli[0], velj[1]-veli[1], velj[2]-veli[2]]
          
          #print 'velj'+str(veli)
          #spiky gradient, inserted in eq 15
          grad = spikyGrad(posi, posJ, h)

          #cross poduct of gradient and vij
          crossProduct = [(vij[1]*grad[2])-vij[2]*grad[1], -(vij[0]*grad[2]-vij[2]*grad[0]), (vij[0]*grad[1])-vij[1]*grad[0]]
        vorticityVec.append(crossProduct)
    #vorticityVec = crossProduct
    vorticityVec.insert(0,[])
    return vorticityVec

def applyXSPH(c, h, Pos, Vel, neighbours,numOfParticles):
    vNew = []
    sumJ = []
    for i in range (1, numOfParticles):
        posI = Pos[i]
        velI = Vel[i]
        for j in range (1, len(neighbours[i])):
              cmds.select(ListOfParticles[Neighbours[i][j]])
              posJ =  [Pos[Neighbours[i][j]][0], Pos[Neighbours[i][j]][1], Pos[Neighbours[i][j]][2]]
              velJ = [Vel[Neighbours[i][j]][0],Vel[Neighbours[i][j]][1],Vel[Neighbours[i][j]][2]]
              
              W = poly6Kernel(posI,posJ, h)
              vIJ = subVect(velI,velJ)

              sumJ += scalarMult(vIJ,W)
              
        sumJ = scalarMult(sumJ,c)
        vNew.append(addVect(velI,sumJ))
    vNew.insert(0,[])
    return vNew
    

#compute corrective force eq 16
def fVorticity(vorticity, particlePosition, epsilon, h,Neighbours):
    fVorticity = []
    
    for i in range (1,numOfParticles):
        posi = particlePosition[i]
        res = [0,0,0]
        
        for j in range (1, len(Neighbours[i])):
        
          vort = vorticity[Neighbours[i][j]]
          vortLen = getLengthOfVec(vort)
          cmds.select(ListOfParticles[Neighbours[i][j]])
          posJ = [cmds.getAttr(".translateX"),cmds.getAttr(".translateY"),cmds.getAttr(".translateZ")]
          grad = spikyGrad(posi, posJ, h)
          
          n = scalarMult(grad,vortLen)
         
          res = addVect(res,n)
          resLength = getLengthOfVec(res)
          
          
          #print nNormFactor
          if resLength < 0.000001:
             fVorticity.append([0,0,0])
             continue
             
          res = norm(res)
          
          crossProduct = [(res[1]*vort[2])-res[2]*vort[1], -(res[0]*vort[2]-res[2]*vort[0]), (res[0]*vort[1])-res[1]*vort[0]]
          crossProduct = scalarMult(crossProduct,0.01)
          fVorticity.append(crossProduct)
    fVorticity.insert(0,[])
    return fVorticity

#Simulation Loop
#Constants
dt = 0.016
MaxSolverIterations = 5
ZeroRho = 1000.0
h = 1.0
c = 0.01
epsilon = 200.0
correctionK = 0.001
correctionN = 4.0
correctionDeltaQ = 0.3

numOfParticles = count+1

#Animation
KeyFrames = 25
cmds.playbackOptions( playbackSpeed = 0, maxPlaybackSpeed = 1, min = 1, max = 150 )
startTime = cmds.playbackOptions( query = True, minTime = True )
endTime = cmds.playbackOptions( query = True, maxTime = True )
frame = startTime

var = 1
particleVelocity = [0]
Neighbours = [0]
PredictedPosition=[0]
Lambda = [0]
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
    cmds.select('Particle' + str(i))
    
    pos = [cmds.getAttr(".translateX"), cmds.getAttr(".translateY"),cmds.getAttr(".translateZ")]
   
   #Assign beginning vals
    PredictedPosition.append([pos[0],pos[1],pos[2]])
    particleVelocity.append([0,0,0])
   
    cmds.setKeyframe(".translateX", value=pos[0], time=frame)
    cmds.setKeyframe(".translateY", value=pos[1], time=frame)
    cmds.setKeyframe(".translateZ", value=pos[2], time=frame)

#Simulation loop
for j in range (1,KeyFrames):
    print 'Frame: ' + str(frame)
    frame += 1
    
    ##OK
    #Predict position and velocities
    for i in range (1,numOfParticles):  
        particleVelocity[i][1] = particleVelocity[i][1] - (dt*9.82)
        PredictedPosition[i][1]= PredictedPosition[i][1] + (dt*particleVelocity[i][1])
            
    #Ok    
    #Find Neighboring particles      
    Neighbours = findNeighboringParticles(numOfParticles,PredictedPosition, h)
    
    
    #Ok according to Muller
    #Iteration loop
    Iter = 0
    while Iter < MaxSolverIterations :
                     
        Lambda = CalculateLambda(numOfParticles, PredictedPosition, Neighbours, ZeroRho, h)
          
        deltaPositions = deltaP(Lambda, ZeroRho, numOfParticles, PredictedPosition, h,Neighbours)
        
        particleCollision = calculateParticleCollisionResponse(PredictedPosition, particleVelocity, particleRadius, Neighbours,numOfParticles)
        
        for i in range (1 , numOfParticles):        
                     PredictedPosition[i][0] = particleCollision[0][i][0]
                     PredictedPosition[i][1] = particleCollision[0][i][1]
                     PredictedPosition[i][2] = particleCollision[0][i][2]
                     particleVelocity[i][0] = particleCollision[1][i][0]
                     particleVelocity[i][1] = particleCollision[1][i][1]
                     particleVelocity[i][2] = particleCollision[1][i][2]
                
        Constraints = BoxConstraints(PredictedPosition,particleVelocity,particleRadius,numOfParticles,var)
                
        for i in range (1 , numOfParticles):
                     
                     PredictedPosition[i][0] = Constraints[0][i][0]
                     PredictedPosition[i][1] = Constraints[0][i][1]
                     PredictedPosition[i][2] = Constraints[0][i][2]
                     particleVelocity[i][0] = Constraints[1][i][0]
                     particleVelocity[i][1] = Constraints[1][i][1]
                     particleVelocity[i][2] = Constraints[1][i][2]
            
        for i in range (1 , numOfParticles):
            PredictedPosition[i][0] +=deltaPositions[i][0]
            PredictedPosition[i][1] +=deltaPositions[i][1]
            PredictedPosition[i][2] +=deltaPositions[i][2]
        Iter +=1
        
      

    
    #End while Loop    
    #print  particleVelocity       
    
    for n in range (1,numOfParticles):
        cmds.select(ListOfParticles[n])
        pos = [cmds.getAttr(".translateX"), cmds.getAttr(".translateY"),cmds.getAttr(".translateZ")]
        particleVelocity[n] = scalarMult(subVect(PredictedPosition[n], pos), (1/dt))
    
    #vort = vorticityConfinement(particleVelocity, PredictedPosition, Neighbours, h, numOfParticles)
    #f_Vorticity = fVorticity(vort, PredictedPosition, epsilon, h,Neighbours)
    XSPH = applyXSPH(c, h, PredictedPosition, particleVelocity, Neighbours,numOfParticles)
                
    for i in range (1, numOfParticles) :
        particleVelocity[i] =  XSPH[i]
        cmds.select( 'Particle'+str(i) )
        cmds.setKeyframe(".translateX", value=PredictedPosition[i][0], time=frame)
        cmds.setKeyframe(".translateY", value=PredictedPosition[i][1], time=frame)
        cmds.setKeyframe(".translateZ", value=PredictedPosition[i][2], time=frame)
    if j <= KeyFrames/2:
        var = var - 0.05
    else:            
        var = var + 0.05
    
print 'Frame: ' + str(KeyFrames)