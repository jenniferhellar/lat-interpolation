
import os

dirName = 'data'

fileList = os.listdir(dirName)

for fileName in fileList:
	lst = fileName.split(' ')
	lst = [i.strip() for i in lst if i.strip() != '']
	newName = '-'.join(lst)
	os.rename(os.path.join(dirName, fileName), os.path.join(dirName, newName))

fileList = os.listdir(dirName)

meshFiles = [i for i in fileList if '.mesh' in i]
latFiles = [i for i in fileList if '_car.txt' in i]
ablFiles = [i for i in fileList if 'Ablation' in i]

numMeshes = len(meshFiles)

for idx in range(numMeshes):
	meshFile = meshFiles[idx]
	latFile = latFiles[idx]

	patientID = latFile[0:10]
	latID = latFile.split('_')[3].split('-')[0]

	ablFile = patientID + '_I_AblationData_' + latID + '.txt'
	if ablFile not in ablFiles:
		ablFile = ''

	meshFile = "'" + meshFile + "'"
	latFile = "'" + latFile + "'"
	ablFile = "'" + ablFile + "'"

	str = '(' + meshFile + ', ' + latFile + ', ' + ablFile + ')'
	if idx == 0:
		str = '[' + str
	if idx < numMeshes-1:
		str = str + ','
	else:
		str = str + ']'
	
	print(str)