import tensorflow as tf
import matplotlib.pyplot as plt
from numpy import loadtxt
import numpy as np
import datetime
from sklearn.metrics import confusion_matrix


dataset_teste = loadtxt("/home/thassio/Desktop/TCC/megasettreino/setTeste4.txt", delimiter='	')

model = tf.keras.models.load_model("/home/thassio/Desktop/modelos/modelo4.h5")

Entradas_Teste = dataset_teste[:,0:5]
Saidas_Teste  = dataset_teste[:,5:11]

Entradas_Teste = tf.keras.utils.normalize(Entradas_Teste, axis=1)

labels = ['Acordado', 'N1','N2','N3','R']


acerto = 0
diferente = 0

prediction = model.predict(Entradas_Teste)

for i in range(193):
	
	if np.argmax(prediction[i]) == np.argmax(Saidas_Teste[i]):
		acerto = acerto + 1
	else:
		diferente = diferente + 1

print('acerto:')
print(acerto)
print('\n')
print('erro:')
print(diferente)

acc2 = acerto/(acerto+diferente)
print('\n')
print('acuracia real:')
print(acc2)

a = []
b = []
for i in range(193):
	a.append(np.argmax(Saidas_Teste[i]))

	b.append(np.argmax(prediction[i]))

cm = confusion_matrix(a, b)

# Show confusion matrix in a separate window

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm, interpolation='nearest')
fig.colorbar(cax)
plt.title('Matriz Confusão - 4º Modelo - Paciente 4')
plt.ylabel('Valor Real')
plt.xlabel('Valor Predito')
ax.set_xticklabels(['']+labels)
ax.set_yticklabels(['']+labels)
for (i, j), z in np.ndenumerate(cm):
    ax.text(j, i, '{:d}'.format(z), ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
plt.show()



conta = 0
hora = 1
estagio = [1]*193
estagiori = [1]*193
compara = [1]*193
labelx = ['‏‏‎ ‎']*193

for i in range(193):
	
	if np.argmax(prediction[i]) == 0:
		estagio[i] = 4
	if np.argmax(prediction[i]) == 1:
		estagio[i] = 2
	if np.argmax(prediction[i]) == 2:
		estagio[i] = 1
	if np.argmax(prediction[i]) == 3:
		estagio[i] = 0
	if np.argmax(prediction[i]) == 4:
		estagio[i] = 3

	if np.argmax(Saidas_Teste[i]) == 0:
		estagiori[i]  = 4
	if np.argmax(Saidas_Teste[i]) == 1:
		estagiori[i]  = 2
	if np.argmax(Saidas_Teste[i]) == 2:
		estagiori[i]  = 1
	if np.argmax(Saidas_Teste[i]) == 3:
		estagiori[i]  = 0
	if np.argmax(Saidas_Teste[i]) == 4:
		estagiori[i]  = 3
	
	
	if conta == 122:
		conta = 0
		labelx[i] = str(hora) + ":00"
		hora = hora +1

	conta = conta +1
	compara[i] = i
labels = ('N3','N2','N1','REM','Acordado')

plt.yticks([0,1,2,3,4], labels)

plt.xticks(compara, labelx)
plt.plot(estagio)
plt.plot(estagiori)

plt.title('Hipnograma - 4º Modelo - Paciente 4')
plt.ylabel('Estágio')
plt.xlabel('Tempo')
plt.legend(['Estágio','Estagio Real'], loc='upper left')
plt.show()