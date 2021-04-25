# face detection for the 5 Celebrity Faces Dataset
from os import listdir
from os.path import isdir
import glob
from PIL import Image
from matplotlib import pyplot
from numpy import savez_compressed
from numpy import asarray
from mtcnn.mtcnn import MTCNN

# extract a single face from a given photograph
def extract_face(detector, filename, required_size=(160, 160)):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = asarray(image)
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    image.save(filename + ".tn.jpeg", "JPEG")

    face_array = asarray(image)
    return face_array


# load images and extract faces for all images in a directory
def load_faces(detector, directory):
    faces = []
    labels = []
    # enumerate files
    print('looking in dir=', directory)
    for filename in glob.glob(directory + '*.jpg'):
        print ('found file: %s' % filename)
        # path
#        path = directory + filename
        # get face
        face = extract_face(detector, filename)
        # store
        faces.append(face)
        labels.append(filename)
    return (faces, labels)

# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
    # create the detector, using default weights
    detector = MTCNN()

    X, y = list(), list()
    # enumerate folders, on per class
    for subdir in listdir(directory):
        print ('found subdir: %s' % subdir )
        # path
        path = directory + subdir + '/'
        # skip any files that might be in the dir
        if not isdir(path):
            continue
        # load all faces in the subdirectory
        (faces, labels) = load_faces(detector, path)
        # create labels
        #labels = [subdir for _ in range(len(faces))]
        # summarize progress
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        # store
        X.extend(faces)
        y.extend(labels)
    return asarray(X), asarray(y)

# load train dataset
trainX, trainY = load_dataset('Rishi/')
print(trainX.shape, trainY.shape)
# load test dataset
#    testX, testy = load_dataset('kaggle/val/')
# save arrays to one file in compressed format
savez_compressed('4-rishi-family-thumbnails.npz', trainX, trainY)
