import os, os.path
from os import scandir
from PIL import Image, ImageDraw, ImageFont
import face_recognition
import numpy as np
import pickle
from flaskapp.models import Person

path = "./flaskapp/static/images/dataset/"

def scantree(path):
    """Recursively yield DirEntry objects for given directory."""
    for entry in scandir(path):
        if entry.is_dir(follow_symlinks=False):
            yield from scantree(entry.path)
        else:
            yield entry


####################################################################
#           Save all face encodings to dataset_faces.dat           #
####################################################################
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
        with open('./flaskapp/dataset_faces.dat', 'wb') as f:
            pickle.dump(all_face_encodings, f)


####################################################################
#                      Register faces in image                     #
####################################################################
def hasSingleFace(image):
	# Load the jpg file into a numpy array
	image = face_recognition.load_image_file(image)

	# Find all the faces in the image using the default HOG-based model.
	# This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
	# See also: find_faces_in_picture_cnn.py
	face_locations = face_recognition.face_locations(image)

	# Find all the faces in the image using a pre-trained convolutional neural network.
	# This method is more accurate than the default HOG model, but it's slower
	# unless you have an nvidia GPU and dlib compiled with CUDA extensions. But if you do,
	# this will use GPU acceleration and perform well.
	# See also: find_faces_in_picture.py
	# face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")

	print("I found {} face(s) in this photograph.".format(len(face_locations)))

	# Return face image only if there is only one face in the image
	if len(face_locations) == 1:
		# Print the location of each face in this image
		top, right, bottom, left = face_locations[0]
		print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

		# You can access the actual face itself like this:
		face_image = image[top:bottom, left:right]
		pil_image = Image.fromarray(face_image)

		return pil_image
	else:
		return None


####################################################################
#                       Detect faces in image                      #
####################################################################
def detectFaces(image):
	# Load the jpg file into a numpy array
	image = face_recognition.load_image_file(image)

	# Find all the faces in the image using the default HOG-based model.
	# This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
	# See also: find_faces_in_picture_cnn.py
	face_locations = face_recognition.face_locations(image)

	# Find all the faces in the image using a pre-trained convolutional neural network.
	# This method is more accurate than the default HOG model, but it's slower
	# unless you have an nvidia GPU and dlib compiled with CUDA extensions. But if you do,
	# this will use GPU acceleration and perform well.
	# See also: find_faces_in_picture.py
	# face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")

	print("I found {} face(s) in this photograph.".format(len(face_locations)))

	# Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
	# See http://pillow.readthedocs.io/ for more about PIL/Pillow
	pil_image = Image.fromarray(image)
	# Create a Pillow ImageDraw Draw instance to draw with
	draw = ImageDraw.Draw(pil_image)

	for face_location in face_locations:
	    # Print the location of each face in this image
	    top, right, bottom, left = face_location
	    print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

	    # You can access the actual face itself like this:
	    # face_image = image[top:bottom, left:right]
	    # pil_image = Image.fromarray(face_image)
	    
	    # Draw a box around the face using the Pillow module
	    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255), width=3)

	# Remove the drawing library from memory as per the Pillow docs
	del draw

	# Display the resulting image
	# pil_image.show()
	return pil_image


####################################################################
#                      Recognize faces in image                    #
####################################################################
def recognizeFaces(image):
	# Get the current working directory (cwd)
	cwd = os.getcwd()
	# Get all the files in that directory
	files = os.listdir(cwd)
	print("Files in '%s': %s" % (cwd, files))
	# Load face encodings
	with open('./flaskapp/dataset_faces.dat', 'rb') as f:
		all_known_face_encodings = pickle.load(f)

	# Grab the list of names and the list of encodings
	known_face_names = list(all_known_face_encodings.keys())
	known_face_encodings = np.array(list(all_known_face_encodings.values()))

	# Load an image with an unknown face
	unknown_image = face_recognition.load_image_file(image)

	# Find all the faces and face encodings in the unknown image
	face_locations = face_recognition.face_locations(unknown_image)
	face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

	# Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
	# See http://pillow.readthedocs.io/ for more about PIL/Pillow
	pil_image = Image.fromarray(unknown_image)
	# Create a Pillow ImageDraw Draw instance to draw with
	draw = ImageDraw.Draw(pil_image)

	# Loop through each face found in the unknown image
	for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
		# See if the face is a match for the known face(s)
		matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

		name = "Unknown"

		# # If a match was found in known_face_encodings, just use the first one.
		# if True in matches:
		#     first_match_index = matches.index(True)
		#     id = known_face_names[first_match_index]
		#     name = Person.query.get(id).name

		# Or instead, use the known face with the smallest distance to the new face
		face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
		best_match_index = np.argmin(face_distances)
		if matches[best_match_index]:
			# add 1 beacuse sqlalchemy starts id from 1
		    id = int(known_face_names[best_match_index])
		    name = Person.query.get(id).name

		# Draw a box around the face using the Pillow module
		draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

		fontsize = 1  # starting font size

		# portion of image width you want text width to be
		img_fraction = 0.98

		font = ImageFont.truetype("./app/static/fonts/GoogleSans-Regular.ttf", fontsize)
		while font.getsize(name)[0] < img_fraction * (right - left):
		    # iterate until the text size is just larger than the criteria
		    fontsize += 1
		    font = ImageFont.truetype("./flaskapp/static/fonts/GoogleSans-Regular.ttf", fontsize)

		# optionally de-increment to be sure it is less than criteria
		fontsize -= 1
		font = ImageFont.truetype("./flaskapp/static/fonts/GoogleSans-Regular.ttf", fontsize)
		# Draw a label with a name below the face
		text_width, text_height = draw.textsize(name)
		draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255), width=3)
		draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255), font=font)

	# Remove the drawing library from memory as per the Pillow docs
	del draw

	# Display the resulting image
	# pil_image.show()
	return pil_image