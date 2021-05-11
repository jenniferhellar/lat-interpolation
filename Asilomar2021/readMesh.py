
"""
--------------------------------------------------------------------------------
Text reader for CARTO exported mesh data.
--------------------------------------------------------------------------------
Description: Reads a triangular mesh from the CARTO .mesh file and returns a
numpy array of the vertices (X, Y, Z) coordinates and a list of triangular faces
(triplets of vertex indices).

File contents:
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

Requirements: numpy

Author: Jennifer Hellar
Email: jennifer.hellar@rice.edu
--------------------------------------------------------------------------------
"""
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
					hFlag = True 	# reached header for vertex section
			if (hFlag):
				cnt = cnt + 1
				if (cnt == 4):
					hFlag = False
					vFlag = True 	# finished header, reached vertex data
			if (vFlag):
				if (line == '[TrianglesSection]'):
					cnt = 0
					vFlag = False
					h1Flag = True 	# reached header for triangles section
				else:
					if (line != ''):
					# 	print(line)
						[X,Y,Z,normalX,normalY,normalZ,groupID] = map(float,line.split()[2:])
						coordinateMatrix.append([X,Y,Z])	# vertex coordinates
			if (h1Flag):
				cnt = cnt + 1
				if (cnt == 4):
					h1Flag = False
					tFlag = True 	# finished header, reached triangles data
			if (tFlag):
				if (line == '[VerticesAttributesSection]'):
					tFlag = False	# finished triangles data
				else:
					if (line != ''):
						# print(line)
						[V0,V1,V2,normalX,normalY,normalZ,groupID] = map(float,line.split()[2:])
						if groupID != -1000000:
							connectivityMatrix.append([V0,V1,V2])	# triangles

	return [np.array(coordinateMatrix), connectivityMatrix]
