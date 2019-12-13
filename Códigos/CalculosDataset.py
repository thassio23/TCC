import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack

#Carregar valores para o código
quantidadedelinhas = 1516
NumeroDeLinhas =  quantidadedelinhas*3000               #Número de amostras por paciente
ColunaSelecionada = 2   #0 Hora / 1 - Canal tem que ver / 2 - Canal tem que ver 

f = open("/home/thassio/Desktop/TCC/sc4112Texto/sc4112e0rec_data.txt","r+")
f1 = open("/home/thassio/Desktop/TCC/sc4112Texto/sc4112e0hyp_data.txt","r+")

linha = []
linha1 = []

AmostrasDe30Segundos = []

for x in range (0, NumeroDeLinhas): #Cria linhas da matriz
	linha.append([])

for x in range(0, NumeroDeLinhas):  #Atribui a Matriz os valores dentro dos arquivos
	linha[x] = f.readline().split(",") #Separa entre colunas os valores de cada linha

for x in range (0, quantidadedelinhas): #Cria linhas da matriz
	linha1.append([])

for x in range(0, quantidadedelinhas):  #Atribui a Matriz os valores dentro dos arquivos
	linha1[x] = f1.readline().split(",") #Separa entre colunas os valores de cada linha

for x in range (0, quantidadedelinhas): #Cria linhas da matriz de amostras
	AmostrasDe30Segundos.append([])

fim = 0

for x in range (0, quantidadedelinhas):
	for y in range (0, 3000):
		AmostrasDe30Segundos[x].append(float(linha[y+fim][ColunaSelecionada]))  #Faz uma matriz quantidadedelinhas linhas para amostras diferentes
	fim = fim + 3000


#Calculo da Transformada de Fourier e dataset

FrequenciaDeAmostragem = 100
Dataset = []

for x in range (0, quantidadedelinhas): #Cria matriz do dataset
	Dataset.append([])


for x in range (0, quantidadedelinhas): #Cria matriz do dataset

	SinalAnalisado = AmostrasDe30Segundos[x]
	QuadradodoSinal = np.square(SinalAnalisado)
	PotenciaMediadoSinal = (np.sum(QuadradodoSinal)/quantidadedelinhas)
	X = fftpack.fft(SinalAnalisado)
	freqs = fftpack.fftfreq(len(SinalAnalisado)) * FrequenciaDeAmostragem
	Dataset[x].append(PotenciaMediadoSinal)
	Dataset[x].append(np.abs(X)[1501:1621].sum())
	Dataset[x].append(np.abs(X)[1621:1741].sum())
	Dataset[x].append(np.abs(X)[1741:1861].sum())
	Dataset[x].append(np.abs(X)[1861:2341].sum())

#file = open("/home/thassio/Desktop/TCC/sc4012Texto/sc4012e0_DataSet.txt","a+") 
file1 = open("/home/thassio/Desktop/TCC/megaset4/set1.txt","a+") 
file2 = open("/home/thassio/Desktop/TCC/megaset4/set2.txt","a+")
file3 = open("/home/thassio/Desktop/TCC/megaset4/set3.txt","a+") 
file4 = open("/home/thassio/Desktop/TCC/megaset4/set3.txt","a+") 
file5 = open("/home/thassio/Desktop/TCC/megaset4/set4.txt","a+") 

for x in range (0, quantidadedelinhas):
	if (str(linha1[x][1]) == "0.000000\n"):
		file1.write(str(round(Dataset[x][0], 4)))
		file1.write("	") 
		file1.write(str(round(Dataset[x][1], 4)))
		file1.write("	") 
		file1.write(str(round(Dataset[x][2], 4)))
		file1.write("	")  
		file1.write(str(round(Dataset[x][3], 4))) 
		file1.write("	") 
		file1.write(str(round(Dataset[x][4], 4)))
		file1.write("	") 
		file1.write(str("1	0	0	0	0\n"))

	if (str(linha1[x][1]) == "1.000000\n"):
		file2.write(str(round(Dataset[x][0], 4)))
		file2.write("	") 
		file2.write(str(round(Dataset[x][1], 4)))
		file2.write("	") 
		file2.write(str(round(Dataset[x][2], 4)))
		file2.write("	")  
		file2.write(str(round(Dataset[x][3], 4))) 
		file2.write("	") 
		file2.write(str(round(Dataset[x][4], 4)))
		file2.write("	") 
		file2.write(str("0	1	0	0	0\n"))

	if (str(linha1[x][1]) == "2.000000\n"):
		file3.write(str(round(Dataset[x][0], 4)))
		file3.write("	") 
		file3.write(str(round(Dataset[x][1], 4)))
		file3.write("	") 
		file3.write(str(round(Dataset[x][2], 4)))
		file3.write("	")  
		file3.write(str(round(Dataset[x][3], 4))) 
		file3.write("	") 
		file3.write(str(round(Dataset[x][4], 4)))
		file3.write("	") 
		file3.write(str("0	0	1	0	0\n"))

	if (str(linha1[x][1]) == "3.000000\n"):
		file4.write(str(round(Dataset[x][0], 4)))
		file4.write("	") 
		file4.write(str(round(Dataset[x][1], 4)))
		file4.write("	") 
		file4.write(str(round(Dataset[x][2], 4)))
		file4.write("	")  
		file4.write(str(round(Dataset[x][3], 4))) 
		file4.write("	") 
		file4.write(str(round(Dataset[x][4], 4)))
		file4.write("	") 
		file4.write(str("0	0	0	1	0\n"))

	if (str(linha1[x][1]) == "4.000000\n"):
		file4.write(str(round(Dataset[x][0], 4)))
		file4.write("	") 
		file4.write(str(round(Dataset[x][1], 4)))
		file4.write("	") 
		file4.write(str(round(Dataset[x][2], 4)))
		file4.write("	")  
		file4.write(str(round(Dataset[x][3], 4))) 
		file4.write("	") 
		file4.write(str(round(Dataset[x][4], 4)))
		file4.write("	") 
		file4.write(str("0	0	0	1	0\n"))
	if (str(linha1[x][1]) == "5.000000\n"):
		file5.write(str(round(Dataset[x][0], 4)))
		file5.write("	") 
		file5.write(str(round(Dataset[x][1], 4)))
		file5.write("	") 
		file5.write(str(round(Dataset[x][2], 4)))
		file5.write("	")  
		file5.write(str(round(Dataset[x][3], 4))) 
		file5.write("	") 
		file5.write(str(round(Dataset[x][4], 4)))
		file5.write("	")
		file5.write(str("0	0	0	0	1\n"))