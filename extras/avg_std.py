
import numpy

NN_NMSE = [0.5414, 0.5416, 0.5026, 0.5411, 0.5369]
NN_SNR = [5.3294, 5.3270, 5.9760, 5.3347, 5.4025]

print('NN')
NN_NMSE_mean = numpy.average(NN_NMSE)
NN_NMSE_std = numpy.std(NN_NMSE)
print('NMSE {:.2f} +/- {:.3f}'.format(NN_NMSE_mean, NN_NMSE_std))

NN_SNR_mean = numpy.average(NN_SNR)
NN_SNR_std = numpy.std(NN_SNR)
print('SNR {:.2f} +/- {:.3f}'.format(NN_SNR_mean, NN_SNR_std))

print('\n')

GPR_NMSE = [0.3082, 0.3142, 0.3135, 0.3127, 0.3091]
GPR_SNR = [10.2245, 10.0550, 10.0758, 10.0968, 10.1987]

print('GPR')
GPR_NMSE_mean = numpy.average(GPR_NMSE)
GPR_NMSE_std = numpy.std(GPR_NMSE)
print('NMSE {:.2f} +/- {:.3f}'.format(GPR_NMSE_mean, GPR_NMSE_std))

GPR_SNR_mean = numpy.average(GPR_SNR)
GPR_SNR_std = numpy.std(GPR_SNR)
print('SNR {:.2f} +/- {:.3f}'.format(GPR_SNR_mean, GPR_SNR_std))

print('\n')

SSL_Graph_NMSE = [0.3023, 0.3063, 0.3045, 0.3036, 0.3003]
SSL_Graph_SNR = [10.3899, 10.2777, 10.3297, 10.3551, 10.4482]

print('SSL_Graph')
SSL_Graph_NMSE_mean = numpy.average(SSL_Graph_NMSE)
SSL_Graph_NMSE_std = numpy.std(SSL_Graph_NMSE)
print('NMSE {:.2f} +/- {:.3f}'.format(SSL_Graph_NMSE_mean, SSL_Graph_NMSE_std))

SSL_Graph_SNR_mean = numpy.average(SSL_Graph_SNR)
SSL_Graph_SNR_std = numpy.std(SSL_Graph_SNR)
print('SNR {:.2f} +/- {:.3f}'.format(SSL_Graph_SNR_mean, SSL_Graph_SNR_std))

print('\n')

SSL_NNGraph_NMSE = [0.0645, 0.0662, 0.0569, 0.0571, 0.0569]
SSL_NNGraph_SNR = [23.8087, 23.5826, 24.8927, 24.8689, 24.8925]

print('SSL_NNGraph')
SSL_NNGraph_NMSE_mean = numpy.average(SSL_NNGraph_NMSE)
SSL_NNGraph_NMSE_std = numpy.std(SSL_NNGraph_NMSE)
print('NMSE {:.2f} +/- {:.3f}'.format(SSL_NNGraph_NMSE_mean, SSL_NNGraph_NMSE_std))

SSL_NNGraph_SNR_mean = numpy.average(SSL_NNGraph_SNR)
SSL_NNGraph_SNR_std = numpy.std(SSL_NNGraph_SNR)
print('SNR {:.2f} +/- {:.3f}'.format(SSL_NNGraph_SNR_mean, SSL_NNGraph_SNR_std))


print('\n')

# for 87 training samples
print(numpy.average([0.0843, 0.0967, 0.0879, 0.0834, 0.0857]))
print(numpy.average([21.4849, 20.2911, 21.1208, 21.5741, 21.3423]))
print('\n')

# for 109 training samples
print(numpy.average([0.0723, 0.0845, 0.0781, 0.0765, 0.0822]))
print(numpy.average([22.8226, 21.4611, 22.1495, 22.3250, 21.7028]))
print('\n')

# 145 training samples
print(numpy.average([0.0672, 0.0783, 0.0754, 0.0724, 0.0752]))
print(numpy.average([23.4589, 22.1281, 22.4516, 22.8052, 22.4805]))
print('\n')

# 174 training samples
print(numpy.average([0.0729, 0.0762, 0.0726, 0.0719, 0.0733]))
print(numpy.average([22.7506, 22.3603, 22.7824, 22.8686, 22.6980]))
print('\n')

# 217 training samples
print(numpy.average([0.0704, 0.0748, 0.0676, 0.0653, 0.0711]))
print(numpy.average([23.0487, 22.5208, 23.4057, 23.7014, 22.9626]))
print('\n')

# 289 training samples
print(numpy.average([0.0681, 0.0718, 0.0619, 0.0644, 0.0677]))
print(numpy.average([23.3351, 22.8726, 24.1650, 23.8243, 23.3823]))
print('\n')

# 434 training samples
print(numpy.average([0.0664, 0.0690, 0.0678, 0.0667, 0.0656]))
print(numpy.average([23.5508, 23.2169, 23.3781, 23.5139, 23.6678]))
print('\n')

# 578 training samples
print(numpy.average([0.0662, 0.0678, 0.0576, 0.0590, 0.0643]))
print(numpy.average([23.5883, 23.3818, 24.7985, 24.5826, 23.8340]))
print('\n')

# 650 training samples
print(numpy.average([0.0652, 0.0670, 0.0576, 0.0574, 0.0565]))
print(numpy.average([23.7200, 23.4842, 24.7897, 24.8196, 24.9556]))
print('\n')

# 722 training samples
print(numpy.average([0.0576, 0.0672, 0.0573, 0.0572, 0.0564]))
print(numpy.average([24.7889, 23.4488, 24.8299, 24.8536, 24.9680]))
print('\n')

# 758 training samples
print(numpy.average([0.0575, 0.0663, 0.0570, 0.0571, 0.0569]))
print(numpy.average([24.8040, 23.5714, 24.8797, 24.8646, 24.8975]))
print('\n')

# 780 training samples
print(numpy.average([0.0574, 0.0579, 0.0562, 0.0573, 0.0568]))
print(numpy.average([24.8221, 24.7441, 24.9987, 24.8389, 24.9089]))

print(numpy.std([0.0574, 0.0579, 0.0562, 0.0573, 0.0568]))
print(numpy.std([24.8221, 24.7441, 24.9987, 24.8389, 24.9089]))
print('\n')