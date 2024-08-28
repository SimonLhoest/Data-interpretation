# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 08:21:50 2022

@author: lhoes
"""

import numpy as np
import matplotlib.pyplot as plt

def centre_red(R):
    '''
    Paramètre
    ----------
    R : Matrice

    Returns
    -------
    Rcr : Matrice centrée réduite

    '''
    m,n=R.shape
    Rcr=np.zeros((m,n))
    X=np.zeros((m,n))
    S=np.zeros((m,n))
    for i in range (n):
        X[:,i]= 1/m*np.sum(R[:,i])
    for i in range (n):
        x=0
        for j in range (m):
           x+=(R[j,i]-X[j,i])**2 
        S[:,i]=x/(m-1)
    Rcr=(R-X)/(((m-1)*S)**(1/2))
    return Rcr
    
def approx(R,k):
    '''
    Paramètres
    ----------
    R : Matrice
    
    k : entier, nombre de direction principale

    Returns
    -------
    pR : Matrice des composantes projetées

    '''
    m,n=np.shape(R)
    if k>m-1 : 
        print('Erreur approx : k>m-1')
        return
    pR=np.zeros((k,m))
    U,S,VT=np.linalg.svd(R)
    for i in range (k):
        pR[i,:]=U[:,i]*S[i]**2
    return pR

def correlationdirprinc(R,k):
    '''
    Paramètres
    ----------
    R : Matrice
    
    k : entier, nombre de direction principale

    Returns
    -------
    cor : Matrice des corrélations.

    '''
    m,n = np.shape(R)
    cor = np.zeros((k,n))
    if k>n :
        print('Erreur correlationdirprinc : k>n')
        return
    U,S,VT = np.linalg.svd(R)
    V=VT.T
    Y=R@V
    for i in range (n):
        for j in range (k):
            cor[j,i]=np.corrcoef(Y[:,j],R[:,i])[1,0]
    return cor

def affichagevar(R,labelsligne,labelscolonne):
    '''
    Paramètres
    ----------
    R : Matrice
    
    labelsligne :   Liste ou matrice
                    contenant les labels des lignes
                    
    labelscolonne : Liste ou matrice
                    contenant les labels des colonnes

    Returns
    -------
    Affiche le graphe qui représente les valeurs des variances σ2
    k et le graphe qui représente le pourcentage de l’explication
    de la variance de chaque k−composante principale

    '''
    U,S,VT = np.linalg.svd(R)
    S=S**2
    plt.figure(1)
    plt.clf()
    plt.subplot(1,2,1)
    bc=plt.bar(labelscolonne,S)
    plt.bar_label(bc, labels = np.around(S,6))
    plt.title('Variance des compansantes principales')
    plt.ylabel('Variance')
    plt.xlabel('Composantes')
    plt.subplot(1,2,2)
    plt.title('Participation à la variance totale')
    plt.pie(S,autopct='%1.1f%%',labels=labelscolonne, shadow=True)

def ACP2D(R,labelsligne,labelscolonne):
    '''
    Paramètres
    ----------
    R : Matrice
    
    labelsligne :   Liste ou matrice
                    contenant les labels des lignes
                    
    labelscolonne : Liste ou matrice
                    contenant les labels des colonnes

    Returns
    ------- 
    Affiche la matrice de sortie de approx(R,2) 
    et le graphe de la matrice de sortie de la
    fonction correlationdirprinc(R,2).
    
    '''
    m,n=np.shape(R)
    affichagevar(R,labelsligne,labelscolonne)

    plt.figure(2)
    plt.clf()
    
    plt.subplot(1,2,1)
    plt.title('Analyse en composante principale pour k=2')
    plt.ylabel('Y\u2081')
    plt.xlabel('Y\u2082')
    M=approx(R,2).T
    x=M[:,0]
    y=M[:,1]
    for i in range (m):
        plt.plot(x[i],y[i],'.')
        plt.annotate(labelsligne[i],(x[i],y[i]))
        
    plt.subplot(1,2,2)
    plt.title('Cercle de corrélation')
    plt.grid(True)
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    x=np.linspace(0,630,1000)/100
    plt.plot(np.cos(x),np.sin(x),linestyle='dashed',color='k')
    cor= correlationdirprinc(R,2)
    for i in range (n):
        plt.arrow(0,0,cor[0,i],cor[1,i],length_includes_head=True,head_width=0.1,head_length=0.1,width=0.02,color='royalblue',label=labelscolonne[i])
        plt.annotate(labelscolonne[i], xy=(cor[0,i], cor[1,i]),fontstyle='oblique',fontsize=12)
        
def ACP3D(R,labelsligne,labelscolonne):
    '''
    Paramètres
    ----------
    R : Matrice
    
    labelsligne :   Liste ou matrice
                    contenant les labels des lignes
                    
    labelscolonne : Liste ou matrice
                    contenant les labels des colonnes

    Returns
    ------- 
    Affiche le graphe en 3D des projections sur les plans 
    qui représentela matrice de sortie de approx(R,3) 
    et le graphe de la matrice de sortie de 
    la fonction correlationdirprinc(R,3).
    
    '''
    m,n=np.shape(R)
    affichagevar(R,labelsligne,labelscolonne)
    
    fig = plt.figure(3)
    plt.clf()
    
    ax = fig.add_subplot(121, projection='3d')
    plt.title('Analyse en composantes principales pour k=3')
    M=approx(R,3).T
    x,y,z = M[:,0],M[:,1],M[:,2]
    for i in range (m):
        ax.scatter(xs=x[i],ys=y[i],zs=z[i])
        ax.text(x[i],y[i],z[i],labelsligne[i])
    ax.set_xlabel('Y\u2081')
    ax.set_ylabel('Y\u2082')
    ax.set_zlabel('Y\u2083')
    
    ax2= fig.add_subplot(122, projection='3d')
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    ax2.set_zlim(-2, 2)
    plt.title('Cercle de corrélation et ses projections')
    cor= correlationdirprinc(R,3)
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    c=np.linspace(0,630,1000)/100
    plt.plot(np.cos(c),np.sin(c),zs=-2,zdir='z',linestyle='dashed',color='grey',alpha=0.5)
    plt.plot(np.cos(c),np.sin(c),zs=2,zdir='y',linestyle='dashed',color='grey',alpha=0.5)
    plt.plot(np.cos(c),np.sin(c),zs=-2,zdir='x',linestyle='dashed',color='grey',alpha=0.5)
    ax2.plot_surface(x, y,z, color="lightblue",alpha=0.25)
    for i in range (n):
        x,y,z = [0,cor[0,i]],[0,cor[1,i]],[0,cor[2,i]]
        ax2.plot(x,y,z,color='royalblue')
        ax2.text(x[1],y[1],z[1],labelscolonne[i])
        ax2.plot(x,y,zs=-2,zdir='z',color='r')
        ax2.text(x[1],y[1],-2,labelscolonne[i])
        ax2.plot(x,z,zs=2,zdir='y',color='r')
        ax2.text(x[1],2,z[1],labelscolonne[i])
        ax2.plot(y,z,zs=-2,zdir='x',color='r')
        ax2.text(-2,y[1],z[1],labelscolonne[i])
        

def ACP(R,labelsligne,labelscolonne,k=0,epsilon=10**(-1)):
    '''
    Paramètres
    ----------
    R : Matrice
    
    labelsligne :   Liste ou matrice
                    contenant les labels des lignes
                    
    labelscolonne : Liste ou matrice
                    contenant les labels des colonnes
                    
    k : entier, optional,
        nombre de direction principale. The default is 0.
        
    epsilon :   float, optional
                Marge. The default is 10**(-1).

    Returns
    -------
    Affiche la représentation graphique
    de la fonction correlationdirprinc(R,k)
    
    '''
    m,n = np.shape(R)
    affichagevar(R,labelsligne,labelscolonne)
    
    if k==0:
        U,S,VT=np.linalg.svd(R)
        S=S**2
        for i in range(len(S)-1):
            if S[i]>=1-epsilon and S[i+1]<1-epsilon:
                k=i+1
                break
    if k==0 :
        print('Erreur ACP : k=0')
        return
    cor = correlationdirprinc(R, k)
    figu = plt.figure(4)
    figu.canvas.manager.set_window_title('Représentation de la matrice de corrélation pour k = {}'.format(k))
    figu.clf()
    ax3 = figu.add_subplot(111)
    img = ax3.imshow(cor,extent=[0,n,0,k])
    figu.colorbar(img)
    L=['Y'+str(-x) for x in range (-k,0)]
    L2=[x+0.5 for x in range (k)]
    L3=[x+0.5 for x in range (n)]
    ax3.set_xticks(L3)
    ax3.set_xticklabels(labelscolonne)
    ax3.set_yticks(L2)
    ax3.set_yticklabels(L)