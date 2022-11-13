from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.optimizers import RMSprop
from NearestNeighbor.data_handler import load_datasets

X_train, y_train, X_test, y_test = load_datasets()
X_train = X_train.reshape(-1,3072)
X_test = X_test.reshape(-1,3072)

pca = PCA(0.9)
pca.fit(X_train)

train_img_pca = pca.transform(X_train)
test_img_pca = pca.transform(X_test)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

batch_size = 128
num_classes = 10
epochs = 20

model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(99,)))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# with pca
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(),  metrics=['accuracy'])
history = model.fit(train_img_pca, y_train,batch_size=batch_size,epochs=epochs,verbose=1, validation_data=(test_img_pca, y_test))

# without pca

X_train = np_utils.to_categorical(X_train)
X_test = np_utils.to_categorical(X_test)

model2 = Sequential()
model2.add(Dense(1024, activation='relu', input_shape=(3072,)))
model2.add(Dense(1024, activation='relu'))
model2.add(Dense(512, activation='relu'))
model2.add(Dense(256, activation='relu'))
model2.add(Dense(num_classes, activation='softmax'))

print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(),  metrics=['accuracy'])
history = model.fit(X_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1, validation_data=(X_test, y_test))
