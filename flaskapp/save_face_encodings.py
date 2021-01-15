import pickle
import os, os.path
from os import scandir
import face_recognition

path = "./static/images/dataset/"

def scantree(path):
    """Recursively yield DirEntry objects for given directory."""
    for entry in scandir(path):
        if entry.is_dir(follow_symlinks=False):
            yield from scantree(entry.path)
        else:
            yield entry

def make_new_face_encodings():
    # save face encodings in dataset_faces.dat using pickle
    all_face_encodings = {}

    for file in scantree(path):
        f_name = file.name
        f_person_id = f_name.split("_")[0]
        print(f_name)
        print(f_person_id)
        image = face_recognition.load_image_file(file.path)
        
        face_encodings = face_recognition.face_encodings(image)
        if len(face_encodings) > 0:
            all_face_encodings[f_person_id] = face_encodings[0]

        # save encoding
        with open('dataset_faces.dat', 'wb') as f:
            pickle.dump(all_face_encodings, f)

make_new_face_encodings()