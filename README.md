# 2022 Software Capstone Design
소프트웨어융합 201710374 원종서

## requirements
> * CUDA  v11.6
> * UE4  v4.27

## Results
<img src="./img/result3.gif" width="60%" height="60%"/>    

## 
<img src="./img/result2.gif" width="45%" height="45%"/> <img src="./img/result1.gif" width="45%" height="45%"/>





## SPH Physics Modelling 
**Navier-Stokes Equations**    
<img src="./img/Navier–Stokes equations.jpg" width="50%" height="50%"/>    


**Density:**    
<img src="./img/density.jpg" width="50%" height="50%"/>    
    
      
        
**Pressure:**    
<img src="./img/pressure.jpg" width="50%" height="50%"/>    
     
     
       
**Viscosity:**    
<img src="./img/viscosity.jpg" width="50%" height="50%"/>    


## SPH Update Routine
<img src="./img/SPH_BP.png" width="100%" height="100%"/>    
    
1. Search neighbors by FRNN(Fixed Radius Near Neighbors) algorithm    
2. Compute density
3. Compute forces(pressure, viscosity, gravity)
4. Compute velocity and position form time intergration
5. Update mesh Position

## Rigid-Fluid Coupling Physics Modelling 
<img src="./img/fluid-rigid.jpg" width="40%" height="45%"/>  <img src="./img/RigidBody.jpg" width="40%" height="45%"/> 

## Screen Space Rendering
<img src="./img/ScreenSpaceRendering.png" width="100%" height="100%"/>
