# calculate a face embedding for each face in the dataset using facenet
from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from keras.models import load_model
import os

# get the face embedding for one face
def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]

def get_embedding_runner():
    # load the face dataset
    data = load('4-rishi-family-thumbnails.npz')
    trainX, trainy = data['arr_0'], data['arr_1']
    print('Loaded: ', trainX.shape, trainy.shape)

    # load the facenet model
    model = load_model('kaggle/facenet_keras.h5')
    print('Loaded Model')

    with open('names.tsv', 'w') as f1:
        for name in trainy:
            head_tail = os.path.split(name)
            print(head_tail[0], head_tail[1], file=f1)

    # convert each face in the train set to an embedding
    with open('embeddings.tsv', 'w') as f2:
        newTrainX = list()
        for face_pixels in trainX:
            embedding = get_embedding(model, face_pixels)
            print(embedding, file=f2)
            newTrainX.append(embedding)
        newTrainX = asarray(newTrainX)
        print(newTrainX.shape)

    # save arrays to one file in compressed format
    savez_compressed('4-rishi-embeddings.npz', newTrainX, trainy)

get_embedding_runner()
