#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import networkx as nx
import numpy as np
import itertools as it
import random
import time
import scipy
from matplotlib import pyplot as plt
import fresnel
import os
import PIL


# In[ ]:


#define the kinetic constant for simulation
k_Au=-2.000000
k_Ag=-4.000000
Energy_ratio=0.100000
Energy_ratio2=0.10000
k_on_Au=0.100000
k_on_Ag=1.000000

#define the relative energy ratio
E_Au_Au=0.3275
E_Au_Ag=Energy_ratio*E_Au_Au # the binding of Au-Ag
E_Ag_Ag=Energy_ratio2*E_Au_Au # the binding of Ag-Ag
miu_Au=k_Au*E_Au_Au # chemical potential of Au-Au
miu_Ag=k_Ag*E_Ag_Ag # chemical potential of Ag-Ag
kT=0.0259
beta=1/kT


# In[ ]:


basis=np.array([np.array([0.5,0.5,0]),
                     np.array([-0.5,0.5,0]),
                     np.array([0.5,-0.5,0]),
                     np.array([-0.5,-0.5,0]),
                     np.array([0.5,0,0.5]),
                     np.array([0.5,0,-0.5]),
                     np.array([-0.5,0,0.5]),
                     np.array([-0.5,0,-0.5]),
                     np.array([0,0.5,0.5]),
                     np.array([0,-0.5,0.5]),
                     np.array([0,0.5,-0.5]),
                     np.array([0,-0.5,-0.5]),
                    ])
basis=np.around(basis,7)


# In[ ]:


def coordination_pattern(x,basis,include_center=False):
    """
    To do: obtain the neighbors of an input atom position
    Args:
        x: the position of atoms
        basis: the relative position to the center atom x
    Returns:
        the neighbors of the atoms, excluding the center atoms
    """
    x=np.array(x)
    if include_center==True:
        temp=(x.reshape(-1,1,3)+basis).reshape(-1,3)
        return np.around(np.vstack((x,temp)),7)
    else:
        return np.around((x.reshape(-1,1,3)+basis).reshape(-1,3),7)


# In[ ]:


def addition_event(flag=False):
    
    #addition
    atom_type=random.choices(["Ag","Au"],weights=[k_on_Ag,k_on_Au])[0]
    #sampling strategy 
    site_surface_site_temp=random.choice(site_vacant_list)
    coor_num_Au=site_all.nodes[site_surface_site_temp]["coor_num_Au"]
    coor_num_Ag=site_all.nodes[site_surface_site_temp]["coor_num_Ag"]
    
    #set probability distribution
    if atom_type=="Au":
        fc=len(site_vacant_list)/max((len(site_surface_Au_list)),1e-10)*np.exp(-(beta)*(-miu_Au-E_Au_Au*coor_num_Au-E_Au_Ag*coor_num_Ag))
        accept_probability=1/(1+1/fc)
        flag=np.random.choice([False,True],1,p=[1-accept_probability,accept_probability]).item()
        
    elif atom_type=="Ag":
        fc=len(site_vacant_list)/max((len(site_surface_Ag_list)),1e-10)*np.exp(-(beta)*(-miu_Ag-E_Au_Ag*coor_num_Au-E_Ag_Ag*coor_num_Ag))
        accept_probability=1/(1+1/fc)
        flag=np.random.choice([False,True],1,p=[1-accept_probability,accept_probability]).item()
    
    #main part    
    if flag==False:
        pass
    else:
        #deal with the case by adding a Au atom
        if atom_type=="Au":
            #deal with the center atom first
            site_vacant_list.remove(site_surface_site_temp) #remove the site from the vacant list
            site_atoms_Au_list.append(site_surface_site_temp)

            site_all.nodes[site_surface_site_temp]["site"]="Au" #change the vacant site to atom site
            if coor_num_Au+coor_num_Ag!=max_coor_num: #if the this atom is not saturated,add to the surface atom
                site_surface_Au_list.append(site_surface_site_temp)
                
            #deal with the neighbor
            coor_sites=coordination_pattern(site_surface_site_temp,basis,include_center=False)
            for coor_site in coor_sites:
                coor_site=tuple(coor_site)
                # if this neighbor dosen't exist in the overall site, it's a vacant site and should be added
                if site_all.has_node(coor_site)==False:
                    site_vacant_list.append(coor_site)
                    site_all.add_node(coor_site)
                    site_all.nodes[coor_site]["coor_num_Au"]=1
                    site_all.nodes[coor_site]["coor_num_Ag"]=0
                    site_all.nodes[coor_site]["site"]="vacant"
                # if this neighbor exists in the overall site, the coordination number is increased by 1.
                # then if it's atom site and saturated, we should delete it from surface atom
                else:
                    site_all.nodes[coor_site]["coor_num_Au"]=site_all.nodes[coor_site]["coor_num_Au"]+1
                    if site_all.nodes[coor_site]["coor_num_Au"]+site_all.nodes[coor_site]["coor_num_Ag"]==max_coor_num and (site_all.nodes[coor_site]["site"]!="vacant"):
                        if site_all.nodes[coor_site]["site"]=="Au":
                            site_surface_Au_list.remove(coor_site)
                        else:
                            site_surface_Ag_list.remove(coor_site)
                            
        #deal with the case by adding a Ag atom
        else:
            #deal with the center atom first
            site_vacant_list.remove(site_surface_site_temp) #remove the site from the vacant list
            site_atoms_Ag_list.append(site_surface_site_temp)

            site_all.nodes[site_surface_site_temp]["site"]="Ag" #change the vacant site to atom site
            
            if coor_num_Au+coor_num_Ag!=max_coor_num: #if the this atom is not saturated,add to the surface atom
                site_surface_Ag_list.append(site_surface_site_temp)
            #deal with the neighbor
            coor_sites=coordination_pattern(site_surface_site_temp,basis,include_center=False)
            for coor_site in coor_sites:
                coor_site=tuple(coor_site)
                # if this neighbor dosen't exist in the overall site, it's a vacant site and should be added
                if site_all.has_node(coor_site)==False:
                    site_vacant_list.append(coor_site)
                    site_all.add_node(coor_site)
                    site_all.nodes[coor_site]["coor_num_Ag"]=1
                    site_all.nodes[coor_site]["coor_num_Au"]=0
                    site_all.nodes[coor_site]["site"]="vacant"
                # if this neighbor exists in the overall site, the coordination number is increased by 1.
                # then if it's atom site and saturated, we should delete it from surface atom
                else:
                    site_all.nodes[coor_site]["coor_num_Ag"]=site_all.nodes[coor_site]["coor_num_Ag"]+1
                    if (site_all.nodes[coor_site]["coor_num_Au"]+site_all.nodes[coor_site]["coor_num_Ag"])==max_coor_num and (site_all.nodes[coor_site]["site"]!="vacant"):
                        if site_all.nodes[coor_site]["site"]=="Au":
                            site_surface_Au_list.remove(coor_site)
                        else:
                            site_surface_Ag_list.remove(coor_site)
    return flag


# In[ ]:


def deletion_event(flag=False):
    #deletion
    if len(site_surface_Ag_list)>0 and len(site_surface_Au_list)>0:
        atom_type=random.choices(["Ag","Au"],weights=[k_on_Ag,k_on_Au])[0]
    elif len(site_surface_Ag_list)>0:
        atom_type="Ag"
    elif len(site_surface_Au_list)>0:
        atom_type="Au"
    
    if atom_type=="Au":
        site_surface_site_temp=random.choice(site_surface_Au_list)
    elif atom_type=="Ag":
        site_surface_site_temp=random.choice(site_surface_Ag_list)
        
    coor_num_Au=site_all.nodes[site_surface_site_temp]["coor_num_Au"]
    coor_num_Ag=site_all.nodes[site_surface_site_temp]["coor_num_Ag"]
    
    if atom_type=="Au":
        #set probability distribution
        fc=len(site_surface_Au_list)/len(site_vacant_list)*np.exp(-(beta)*(miu_Au+E_Au_Au*coor_num_Au+E_Au_Ag*coor_num_Ag))
        accept_probability=1/(1+1/fc)
        flag=np.random.choice([False,True],1,p=[1-accept_probability,accept_probability]).item()
        
    elif atom_type=="Ag":
        #set probability distribution
        fc=len(site_surface_Ag_list)/len(site_vacant_list)*np.exp(-(beta)*(miu_Ag+E_Au_Ag*coor_num_Au+E_Ag_Ag*coor_num_Ag))
        accept_probability=1/(1+1/fc)
        flag=np.random.choice([False,True],1,p=[1-accept_probability,accept_probability]).item()

    #main part
    if flag==False:
        pass
    else:
        if atom_type=="Au":
            #deal with the center atom first
            site_surface_Au_list.remove(site_surface_site_temp) #remove the site from the surface list
            site_atoms_Au_list.remove(site_surface_site_temp)
            site_all.nodes[site_surface_site_temp]["site"]="vacant" #change the surface site to vacant site
            if coor_num_Au+coor_num_Ag!=0: #if this vacant site is attached to any atoms
                site_vacant_list.append(site_surface_site_temp)
            else:
                site_all.remove_node(site_surface_site_temp)
            #deal with the neighbor
            coor_sites=coordination_pattern(site_surface_site_temp,basis,include_center=False)
            for coor_site in coor_sites:
                coor_site=tuple(coor_site)
                # if this neighbor dosen't exist in the overall site, it's a mistake
                if site_all.has_node(coor_site)==False:
                    print("Error!")
                # by deleting the atom, the coordination numbers of the neighbors are all decreased by 1
                else:
                    site_all.nodes[coor_site]["coor_num_Au"]=site_all.nodes[coor_site]["coor_num_Au"]-1
                    # if the coordination number is decrease to 0, meaning a detachment happended, if the site is vacant, it's removed
                    if site_all.nodes[coor_site]["coor_num_Au"]+site_all.nodes[coor_site]["coor_num_Ag"]==0 and site_all.nodes[coor_site]["site"]=="vacant":
                        site_all.remove_node(coor_site)
                        site_vacant_list.remove(coor_site)
                    # it will also make the transform from body atom to surface atom
                    elif site_all.nodes[coor_site]["coor_num_Au"]+site_all.nodes[coor_site]["coor_num_Ag"]==max_coor_num-1 and site_all.nodes[coor_site]["site"]!="vacant":
                        if site_all.nodes[coor_site]["site"]=="Au":
                            site_surface_Au_list.append(coor_site)
                        elif site_all.nodes[coor_site]["site"]=="Ag":
                            site_surface_Ag_list.append(coor_site)   
        else:
            #deal with the center atom first
            site_surface_Ag_list.remove(site_surface_site_temp) #remove the site from the surface list
            site_atoms_Ag_list.remove(site_surface_site_temp)
            site_all.nodes[site_surface_site_temp]["site"]="vacant" #change the surface site to vacant site
            if coor_num_Au+coor_num_Ag!=0: #if this vacant site is attached to any atoms
                site_vacant_list.append(site_surface_site_temp)
            else:
                site_all.remove_node(site_surface_site_temp)
            #deal with the neighbor
            coor_sites=coordination_pattern(site_surface_site_temp,basis,include_center=False)
            
            for coor_site in coor_sites:
                coor_site=tuple(coor_site)
                # if this neighbor dosen't exist in the overall site, it's a mistake
                if site_all.has_node(coor_site)==False:
                    print("Error!")
                # by deleting the atom, the coordination numbers of the neighbors are all decreased by 1
                else:
                    site_all.nodes[coor_site]["coor_num_Ag"]=site_all.nodes[coor_site]["coor_num_Ag"]-1
                    # if the coordination number is decrease to 0, meaning a detachment happended, if the site is vacant, it's removed
                    if site_all.nodes[coor_site]["coor_num_Au"]+site_all.nodes[coor_site]["coor_num_Ag"]==0 and site_all.nodes[coor_site]["site"]=="vacant":
                        site_all.remove_node(coor_site)
                        site_vacant_list.remove(coor_site)
                    # it will also make the transform from body atom to surface atom
                    elif (site_all.nodes[coor_site]["coor_num_Au"]+site_all.nodes[coor_site]["coor_num_Ag"])==(max_coor_num-1) and site_all.nodes[coor_site]["site"]!="vacant":
                        if site_all.nodes[coor_site]["site"]=="Au":
                            site_surface_Au_list.append(coor_site)
                        elif site_all.nodes[coor_site]["site"]=="Ag":
                            site_surface_Ag_list.append(coor_site)
    return flag


# In[ ]:


def replacement_event(flag=False):
    #replacement
    #sampling strategy
    if len(site_surface_Ag_list)>0 and len(site_surface_Au_list)>0:
        atom_type=random.choices(["Au","Ag"],weights=[k_on_Ag,k_on_Au])[0]
    elif len(site_surface_Ag_list)>0:
        atom_type="Ag"
    elif len(site_surface_Au_list)>0:
        atom_type="Au"
    
    if atom_type=="Au":
        site_temp=random.choice(site_surface_Au_list)
    elif atom_type=="Ag":
        site_temp=random.choice(site_surface_Ag_list)
        
    coor_num_Au=site_all.nodes[site_temp]["coor_num_Au"]
    coor_num_Ag=site_all.nodes[site_temp]["coor_num_Ag"]
    
    if atom_type=="Au":
        #set probability distribution
        fc=len(site_surface_Au_list)/max(len(site_surface_Ag_list),1e-10)*np.exp(-(beta)*(miu_Au-miu_Ag+(E_Au_Au-E_Au_Ag)*coor_num_Au+(E_Au_Ag-E_Ag_Ag)*coor_num_Ag))
        accept_probability=1/(1+1/fc)
        flag=np.random.choice([False,True],1,p=[1-accept_probability,accept_probability]).item()
        
    elif atom_type=="Ag":
        #set probability distribution
        fc=len(site_surface_Ag_list)/max(len(site_surface_Au_list),1e-10)*np.exp(-(beta)*(miu_Ag-miu_Au+(E_Au_Ag-E_Au_Au)*coor_num_Au+(E_Ag_Ag-E_Au_Ag)*coor_num_Ag))
        accept_probability=1/(1+1/fc)
        flag=np.random.choice([False,True],1,p=[1-accept_probability,accept_probability]).item()
        
    
    #Main change part
    if flag==False:
        pass
    
    else:
        if atom_type=="Au":
            #deal with the center cell first
            #update list
            site_atoms_Au_list.remove(site_temp)
            site_atoms_Ag_list.append(site_temp)
            if coor_num_Au+coor_num_Ag<max_coor_num:
                site_surface_Au_list.remove(site_temp)
                site_surface_Ag_list.append(site_temp) 
            #update graph
            site_all.nodes[site_temp]["site"]="Ag"
            #deal with the neighbor
            coor_sites=coordination_pattern(site_temp,basis,include_center=False)
            for coor_site in coor_sites:
                coor_site=tuple(coor_site)
                # if this neighbor exists in the overall site, 
                #the coordination number of this site is changed because of replacement
                site_all.nodes[coor_site]["coor_num_Ag"]=site_all.nodes[coor_site]["coor_num_Ag"]+1
                site_all.nodes[coor_site]["coor_num_Au"]=site_all.nodes[coor_site]["coor_num_Au"]-1
        else:
            #deal with the center cell first
            #update list
            site_atoms_Ag_list.remove(site_temp)
            site_atoms_Au_list.append(site_temp)
            if coor_num_Au+coor_num_Ag<max_coor_num:
                site_surface_Ag_list.remove(site_temp)
                site_surface_Au_list.append(site_temp) 
            #update graph
            site_all.nodes[site_temp]["site"]="Au"
            #deal with the neighbor
            coor_sites=coordination_pattern(site_temp,basis,include_center=False)
            for coor_site in coor_sites:
                coor_site=tuple(coor_site)
                # if this neighbor exists in the overall site, 
                #the coordination number of this site is changed because of replacement
                site_all.nodes[coor_site]["coor_num_Au"]=site_all.nodes[coor_site]["coor_num_Au"]+1
                site_all.nodes[coor_site]["coor_num_Ag"]=site_all.nodes[coor_site]["coor_num_Ag"]-1
    
    return flag


# In[ ]:
cpu_limit = fresnel.Device(mode='cpu', n=1)

site_all=nx.Graph()
site_vacant=nx.Graph()
site_surface_Au=nx.Graph()
site_atoms_Au=nx.Graph()

max_coor_num=12
position=np.loadtxt("/home/yibin/workspace/Growth_with_silverV3/atomList_TruncatedOctahedron.csv",delimiter=",")
                
#create the initial shape of a truncated_octahedron
for i in position:
    #if some criteria is satisfied
    site_all.add_node(tuple(i))
    
site_atoms_Au.add_nodes_from(site_all)
site_atoms_Au_list=list(site_atoms_Au)
site_atoms_Ag_list=[]

#calculate the initial surface atoms, vacant sites and corresponding coordination numbers
#for all the atoms
for site_all_list_temp in site_atoms_Au_list:
    #obtain the coordinated atoms to that specific atom
    coor=coordination_pattern(np.asarray(site_all_list_temp),basis)
    #begin to count how many atoms already exsting in the site list
    coor_num=0
    for coor_temp in coor:
        coor_temp=tuple(coor_temp)
        if site_atoms_Au.has_node(coor_temp):
            coor_num=coor_num+1
        else:
            # if not exist, it's a vacant site, so we add it to the overall site and vacant site graph
            if site_vacant.has_node(coor_temp):
                site_all.nodes[coor_temp]["coor_num_Au"]=site_all.nodes[coor_temp]["coor_num_Au"]+1
            else:
                site_all.add_node(tuple(coor_temp))
                site_vacant.add_node(tuple(coor_temp))
                site_all.nodes[coor_temp]["coor_num_Au"]=1
                site_all.nodes[coor_temp]["coor_num_Ag"]=0
                site_all.nodes[coor_temp]["site"]="vacant"

    #record the coordination of that specific atom        
    site_all.nodes[site_all_list_temp]["coor_num_Au"]=coor_num
    site_all.nodes[site_all_list_temp]["coor_num_Ag"]=0
    site_all.nodes[site_all_list_temp]["site"]="Au"

    #if the coordination number is not saturated, it's a surface atom and add to the surface atom graph
    if coor_num<max_coor_num:
        site_surface_Au.add_node(site_all_list_temp)

site_surface_Au_list=list(site_surface_Au)
site_surface_Ag_list=[]
site_vacant_list=list(site_vacant)

path="/mnt/STORAGE2/Nanobot-Monte_Carlo_Simulation/Au_Ag_7/example_{}_{}_{}_{}_{}_{}/".format(k_Au,k_Ag,Energy_ratio,Energy_ratio2,k_on_Au,k_on_Ag)

if os.path.isdir(path)==False:
    os.mkdir(path)
    os.mkdir(path+"structure")
    os.mkdir(path+"plot")

# In[ ]:
#plot the seeds
data=np.array(site_atoms_Au_list+site_atoms_Ag_list)

scene = fresnel.Scene(device=cpu_limit)
geometry1 = fresnel.geometry.Sphere(scene, N=len(data), radius=np.sqrt(2)/4)
geometry1.material = fresnel.material.Material(color=fresnel.color.linear([0.25,0.5,0.9]),
                                              roughness=0.8)

geometry1.position[:] = data
geometry1.material.primitive_color_mix = 1.0

geometry1.color[0:len(site_atoms_Au_list)]=fresnel.color.linear([1,1,0])
geometry1.color[len(site_atoms_Au_list):len(data)]=fresnel.color.linear([0.25,0.5,0.9])

out=fresnel.preview(scene)
image = PIL.Image.fromarray(out[:], mode='RGBA')
image.save(path+"plot/"+"seed.png")

def obtain_coor_number_for_atoms(atom_list,site_all):
    if len(atom_list)>0:
        atom_list_all_data=np.hstack((np.array(atom_list),
                                    np.array([site_all.nodes[coor_site]["coor_num_Au"] for coor_site in atom_list]).reshape(-1,1),
                                    np.array([site_all.nodes[coor_site]["coor_num_Ag"] for coor_site in atom_list]).reshape(-1,1)))
    else:
        atom_list_all_data=[]
    return atom_list_all_data

# In[ ]:
size_original=len(site_surface_Au_list)+len(site_surface_Ag_list)
geometry=[]
len1=size_original

np.random.seed(42)
random.seed(42)

count=0
for count in range(4000001):
    event_index=np.random.choice([0,1,2],1).item()
    if event_index==0:
        flag=addition_event()
    elif event_index==1:
        flag=replacement_event()
    elif event_index==2:
        flag=deletion_event()
        
    if count%500000==0:
        
        Au_data=obtain_coor_number_for_atoms(site_atoms_Au_list,site_all)
        Ag_data=obtain_coor_number_for_atoms(site_atoms_Ag_list,site_all)
        
        np.savetxt(path+"structure/"+"Au_data_%d.csv"%count,Au_data,delimiter=",")
        np.savetxt(path+"structure/"+"Ag_data_%d.csv"%count,Ag_data,delimiter=",")
        
    if count%100000==0:
        data=np.array(site_atoms_Au_list+site_atoms_Ag_list)
        scene = fresnel.Scene(device=cpu_limit)
        geometry1 = fresnel.geometry.Sphere(scene, N=len(data), radius=np.sqrt(2)/4)
        geometry1.material = fresnel.material.Material(color=fresnel.color.linear([0.25,0.5,0.9]),
                                                      roughness=0.8)
        geometry1.position[:] = data
        geometry1.material.primitive_color_mix = 1.0

        geometry1.color[0:len(site_atoms_Au_list)]=fresnel.color.linear([1,1,0])
        geometry1.color[len(site_atoms_Au_list):len(data)]=fresnel.color.linear([0.25,0.5,0.9])

        out=fresnel.preview(scene)
        image = PIL.Image.fromarray(out[:], mode='RGBA')
        image.save(path+"plot/"+"%d.png"%count)