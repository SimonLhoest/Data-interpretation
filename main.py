# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 08:21:40 2022

@author: lhoes
"""

import ACP as acp
import numpy as np
#%% IRIS
Ebrut = np.genfromtxt("iris.csv",dtype=str,delimiter=',') #données brutes
labelscolonne = Ebrut[0,:-1]
labelsligne= Ebrut[1:,-1]
E=Ebrut[1:,:-1].astype('float')
#%%
Ecr=acp.centre_red(E)
acp.ACP(Ecr,labelsligne,labelscolonne)
acp.ACP2D(Ecr,labelsligne,labelscolonne)
acp.ACP3D(Ecr,labelsligne,labelscolonne)

#%% PIZZA
Gbrut = np.genfromtxt('Pizzamod.csv',dtype=str,delimiter=',')
Glabelscolonne= Gbrut[0,1:]
Glabelsligne=Gbrut[1:,0]
G=Gbrut[1:,1:].astype('float')
#%% 
Gcr=acp.centre_red(G)
acp.ACP2D(Gcr,Glabelsligne,Glabelscolonne)

#%% CRANE
Hbrut = np.genfromtxt('Howellmod.csv',dtype=str,delimiter=',')
Hlabelscolonne= Hbrut[0,1:]
Hlabelsligne=Hbrut[1:,0]
H=Hbrut[1:,1:].astype('float')
#%%
Hcr=acp.centre_red(H) #C'est très long
acp.ACP(Hcr,Hlabelsligne,Hlabelscolonne)

#%% Performance en fonction du rôle dans un jeu compétitif. https://www.kaggle.com/jordipompas/lec-regular-season-2021?select=Total+Season+LEC2021.csv
Ibrut = np.genfromtxt('Total Season LEC2021.csv',dtype=str,delimiter=';')[:,:-2]
m,n=Ibrut.shape
Ilabelscolonne = Ibrut[0,6:9]
Ilabelsligne = Ibrut[1:,0]
for i in range(1,m):
    for j in [5,14,15,16]:
        Ibrut[i,j]=Ibrut[i,j].replace('%','')
I=Ibrut[1:,6:9].astype('float')
#%%
Icr=acp.centre_red(I)
acp.ACP(Icr, Ilabelsligne, Ilabelscolonne,k=3)

#%% Performance des élèves en fonction de leur repas (payant ou gratuit/réduit) https://www.kaggle.com/allexanderspb/studentsperformance
Jbrut = np.genfromtxt('StudentsPerformance.csv',dtype=str,delimiter=',')
m,n = Jbrut.shape
for i in range (m):
    for j in range (n):
        Jbrut[i,j]=Jbrut[i,j].replace('"','')
Jlabelscolonne = Jbrut[0,-3:]
Jlabelsligne = Jbrut[1:,3]
J=Jbrut[1:,-3:].astype('float')
#%%
Jcr=acp.centre_red(J)
acp.ACP2D(Jcr,Jlabelsligne,Jlabelscolonne)




