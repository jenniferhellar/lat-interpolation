
"""
readMesh.py
--------------------------------------------------------------------------------
Description: Reads a mesh from the .mesh file
.mesh file contents

1 #TriangulatedMeshVersion2.0
2 ; Biosense Webster Triangulated Mesh file format, 2008
3 ; Rights Biosense Webster, LTD
4  ; http://www.biosensewebster.com
5
6 [GeneralAttributes]
7 MeshID                 = 
8 MeshName               = 
9 NumVertex              = 
10 NumTriangle            = 
11 TopologyStatus         = 
12 MeshColor              = 
13 Matrix                 = 
14 NumVertexColors        = 
15
16 [VerticesSection]
17 ;                   X             Y             Z        NormalX   NormalY   NormalZ  GroupID
18
19 *data of vertices*
		.
		.
		.

   [TrianglesSection]
   ;           Vertex0  Vertex1  Vertex2     NormalX   NormalY   NormalZ  GroupID

   *connectivity of triangles*
		.
		.
		.

   [VerticesAttributesSection]
   ;   EML
   ; EML =0: Vertex is regular vertex (not EML)
   ;     =1: Vertex has a EML attribute

Usage:
	>python plotMesh.py 
	

Modules Used: 

--------------------------------------------------------------------------------
"""
__author__ = "Jennifer Hellar"

import numpy as np

def readMesh(fileName):
	coordinateMatrix = []
	connectivityMatrix = []

	with open(fileName,'r') as fID:
		sFlag = True
		hFlag = False
		vFlag = False
		h1Flag = False
		tFlag = False
		cnt = 0
		for line in fID:
			line = line.strip()
			if (sFlag):
				if (line == '[VerticesSection]'):
					cnt = 0
					sFlag = False
					hFlag = True
			if (hFlag):
				cnt = cnt + 1
				if (cnt == 4):
					hFlag = False
					vFlag = True 
			if (vFlag):
				if (line == '[TrianglesSection]'):
					cnt = 0
					vFlag = False
					h1Flag = True
				else:
					if (line != ''):
					# 	print(line)
						[X,Y,Z,normalX,normalY,normalZ,groupID] = map(float,line.split()[2:])
						coordinateMatrix.append([X,Y,Z])
			if (h1Flag):
				cnt = cnt + 1
				if (cnt == 4):
					h1Flag = False
					tFlag = True
			if (tFlag):
				if (line == '[VerticesAttributesSection]'):
					tFlag = False
				else:
					if (line != ''):
						# print(line)
						[V0,V1,V2,normalX,normalY,normalZ,groupID] = map(float,line.split()[2:])
						if groupID != -1000000:
							connectivityMatrix.append([V0,V1,V2])

	return [np.array(coordinateMatrix), connectivityMatrix]
