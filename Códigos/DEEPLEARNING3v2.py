import tensorflow as tf
import matplotlib.pyplot as plt
from numpy import loadtxt
import numpy as np
import datetime
from sklearn.metrics import confusion_matrix
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint



dataset_treino = loadtxt("/home/thassio/Desktop/TCC/megasettreino/megaset.txt", delimiter='	')
dataset_teste = loadtxt("/home/thassio/Desktop/TCC/megasettreino/setTeste1.txt", delimiter='	')
dataset_val = loadtxt("/home/thassio/Desktop/TCC/megasettreino/setVal.txt", delimiter='	')

Entradas_Treino = dataset_treino[:,0:5]
Saidas_Treino = dataset_treino[:,5:11]

Entradas_Teste = dataset_teste[:,0:5]
Saidas_Teste  = dataset_teste[:,5:11]

Entradas_Val = dataset_val[:,0:5]
Saidas_Val  = dataset_val[:,5:11]

Entradas_Treino = tf.keras.utils.normalize(Entradas_Treino, axis=1)
Entradas_Teste = tf.keras.utils.normalize(Entradas_Teste, axis=1)
Entradas_Val = tf.keras.utils.normalize(Entradas_Val, axis=1)

with tf.device('/gpu:0'):
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Dense(5, input_dim=5))


	model.add(tf.keras.layers.Dense(64))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.Activation('tanh'))


	model.add(tf.keras.layers.Dense(64))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.Activation('elu'))


	model.add(tf.keras.layers.Dense(64))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.Activation('tanh'))
	model.add(tf.keras.layers.Dropout(0.5))

	model.add(tf.keras.layers.Dense(5))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.Activation('softmax'))
	print(model.summary())
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	mcp_save = ModelCheckpoint("/home/thassio/Desktop/modelos/modelo5.h5", save_best_only=True, monitor='val_acc', mode='max')

	history = model.fit(Entradas_Treino, Saidas_Treino,epochs=10, validation_data=(Entradas_Val, Saidas_Val), batch_size=8, callbacks=[mcp_save])


#model.save("/home/thassio/Desktop/modelos/modelo5.h5")

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Acurácia - 4º Modelo')
plt.ylabel('Acurácia')
plt.xlabel('Época')
plt.legend(['Acurácia - Treino', 'Acurácia - Teste'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Perda - 4º Modelo')
plt.ylabel('Perda')
plt.xlabel('Época')
plt.legend(['Perda - Treino', 'Perda - Teste'], loc='upper left')
plt.show()


labels = ['Acordado', 'N1','N2','N3','R']


acerto = 0
diferente = 0

prediction = model.predict(Entradas_Teste)

for i in range(191):
	
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
for i in range(191):
	a.append(np.argmax(Saidas_Teste[i]))

	b.append(np.argmax(prediction[i]))

cm = confusion_matrix(a, b)

# Show confusion matrix in a separate window

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm, interpolation='nearest')
fig.colorbar(cax)
plt.title('Matriz Confusão - 4º Modelo')
plt.ylabel('Valor Real')
plt.xlabel('Valor Predito')
ax.set_xticklabels(['']+labels)
ax.set_yticklabels(['']+labels)
for (i, j), z in np.ndenumerate(cm):
    ax.text(j, i, '{:d}'.format(z), ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
plt.show()