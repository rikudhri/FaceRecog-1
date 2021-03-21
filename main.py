import extract_face as face_extraction
import get_embedding as get_embedding
import linear_svm as linear_svm
import predict_face as predict_face
from numpy import savez_compressed

def main():
#    # load the photo and extract the face
#    pixels = face_extraction.extract_face('face.jpg')

      # load train dataset
      trainX, trainy = face_extraction.load_dataset('kaggle/train/')
      print(trainX.shape, trainy.shape)
      # load test dataset
      testX, testy = face_extraction.load_dataset('kaggle/val/')
      # save arrays to one file in compressed format
      savez_compressed('5-celebrity-faces-dataset.npz', trainX, trainy, testX, testy) 


if __name__ == "__main__":
    main()
