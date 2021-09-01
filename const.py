
import os
import platform

mySys = platform.system()

if mySys == 'Linux':
	workDir = '/home/jlh24/latInterpolation'
elif mySys == 'Windows':
	workDir = 'D:\jhell\git-repos\lat-interpolation'
else:
	print('\nERROR: unknown operating system. Must specify workDir and dataDir in const.py\n')
	exit(1)

DATADIR = os.path.join(workDir, 'data')

DATAFILES = [('Patient031_I_MESHData4-SINUS-LVFAM.mesh', 'Patient031_I_LATSpatialData_4-SINUS-LVFAM_car.txt', 'Patient031_I_AblationData_4.txt'),
('Patient032_I_MESHData1-LVFAM-LAT-HYB.mesh', 'Patient032_I_LATSpatialData_1-LVFAM-LAT-HYB_car.txt', 'Patient032_I_AblationData_1.txt'),
('Patient032_I_MESHData2-LVFAM-INITIAL-PVC.mesh', 'Patient032_I_LATSpatialData_2-LVFAM-INITIAL-PVC_car.txt', 'Patient032_I_AblationData_2.txt'),
('Patient032_I_MESHData4-LVFAM-SINUS.mesh', 'Patient032_I_LATSpatialData_4-LVFAM-SINUS_car.txt', 'Patient032_I_AblationData_4.txt'),
('Patient033_I_MESHData3-RV-FAM-PVC-A-NORMAL.mesh', 'Patient033_I_LATSpatialData_3-RV-FAM-PVC-A-NORMAL_car.txt', 'Patient033_I_AblationData_3.txt'),
('Patient033_I_MESHData4-RV-FAM-PVC-A-LAT-HYBRID.mesh', 'Patient033_I_LATSpatialData_4-RV-FAM-PVC-A-LAT-HYBRID_car.txt', 'Patient033_I_AblationData_4.txt'),
('Patient034_I_MESHData4-RVFAM-LAT-HYBRID.mesh', 'Patient034_I_LATSpatialData_4-RVFAM-LAT-HYBRID_car.txt', 'Patient034_I_AblationData_4.txt'),
('Patient034_I_MESHData5-RVFAM-PVC.mesh', 'Patient034_I_LATSpatialData_5-RVFAM-PVC_car.txt', 'Patient034_I_AblationData_5.txt'),
('Patient034_I_MESHData6-RVFAM-SINUS-VOLTAGE.mesh', 'Patient034_I_LATSpatialData_6-RVFAM-SINUS-VOLTAGE_car.txt', 'Patient034_I_AblationData_6.txt'),
('Patient035_I_MESHData8-SINUS.mesh', 'Patient035_I_LATSpatialData_8-SINUS_car.txt', 'Patient035_I_AblationData_8.txt'),
('Patient037_I_MESHData12-LV-SINUS.mesh', 'Patient037_I_LATSpatialData_12-LV-SINUS_car.txt', 'Patient037_I_AblationData_12.txt'),
('Patient037_I_MESHData9-RV-SINUS-VOLTAGE.mesh', 'Patient037_I_LATSpatialData_9-RV-SINUS-VOLTAGE_car.txt', '')]

