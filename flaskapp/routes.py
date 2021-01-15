import secrets
import os
import shutil
from PIL import Image
from flask import render_template, url_for, flash, redirect, Response, request, make_response, send_from_directory
from flaskapp import app, db, bcrypt
from flaskapp.forms import RegistrationForm, LoginForm, UpdateAccountForm, UpdateDatasetForm, SignUpForm
from flaskapp.models import User, Person, Dataset
from flask_login import login_user, current_user, logout_user, login_required
from flaskapp.camera import VideoCamera
from flaskapp import faces

static_folder_path = './flaskapp/static/'

@app.route("/")
@app.route("/home")
@login_required
def home():
    return render_template('home.html', title='home')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/about")
@login_required
def about():
    return render_template('about.html', title='about')


@app.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = SignUpForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash(f'Your account has been created!', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)


@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('home'))
        else:
            flash('Login unsuccessful! Please check login details.', 'danger')
    return render_template('login.html', title='Login', form=form)

@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('home'))

def save_picture(form_picture):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = random_hex + f_ext
    picture_path = os.path.join(app.root_path,'static/profile_pics', picture_fn)
#    form_picture.save(picture_path)
    output_size = (125,125)
    i = Image.open(form_picture)
    i.thumbnail(output_size)
    i.save(picture_path)
    
    return picture_fn

@app.route("/account", methods=['GET','POST'])
@login_required
def account():
    form = UpdateAccountForm()
    if form.validate_on_submit():
        if form.picture.data:
            picture_file = save_picture(form.picture.data)
            current_user.image_file = picture_file
        current_user.username = form.username.data
        current_user.email = form.email.data
        db.session.commit()
        flash('Your account has been updated!','success')
        return redirect(url_for('account'))
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.email.data = current_user.email
    image_file = url_for('static', filename='profile_pics/'+ current_user.image_file)
    return render_template('account.html', title = 'Account', image_file= image_file, form=form)

def save_image_to_dataset(dir_name, form_image_name, to_be_saved_image):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_image_name)
    image_fn = dir_name + '_' + random_hex + f_ext
    try:
        os.makedirs(static_folder_path + 'images/dataset/' + dir_name)
    except FileExistsError:
        print("directory already exists")
        pass
    image_path = os.path.join(app.root_path, 'static/images/dataset/' + dir_name, image_fn)

    output_size = (200, 200)
    i = to_be_saved_image
    i.thumbnail(output_size)
    i.save(image_path)
    
    return image_fn, image_path

@app.route("/faces/register", methods=['GET','POST'])
@login_required
def register_face():
    form = RegistrationForm()
    if form.validate_on_submit():
        if form.images.data:
            # print("##############################################################\n")
            # print(form.images.data)
            images = []
            # random_hex = secrets.token_hex(8)
            name = form.name.data
            person = Person(name=name)
            db.session.add(person)
            db.session.commit()
            id = Person.query.filter_by(name=name).first().id
            has_at_least_one_image_with_single_face = False
            for image in form.images.data:
                # TODO see if there is one only one face in the image (because suppose if there are 
                # 2 persons in the image and the 2nd one tries to recognize himself then if id folder
                # of 1st one comes first than the 2nd one's id folder, 2nd one will be recognized as
                # 1st person as the photo is in 1st person's id folder)
                face_image = faces.hasSingleFace(image)
                if face_image is not None:
                    has_at_least_one_image_with_single_face = True
                    image_fn, image_path = save_image_to_dataset(dir_name=str(id), 
                        form_image_name=image.filename, to_be_saved_image=face_image)
                    dataset = Dataset(image_file=image_fn, author=person)
                    db.session.add(dataset)
                    print(image_path)
                    images.append(image_fn)
            if has_at_least_one_image_with_single_face is True:
                db.session.commit()
                faces.make_new_face_encodings()
                flash(f'Congratulations! Successfully registered the face as {form.name.data}. Try recognizing {form.name.data}.', 'success')
                return redirect(url_for('home'))
            else:
                flash(f'{form.name.data} not registered as there was no face in the image. Try providing different images.', 'danger')
                return render_template('register_face.html', title="Register Face", selectedListElement="registerFace", form=form)
    return render_template('register_face.html', title="Register Face", selectedListElement="registerFace", form=form)


# @app.route("/faces/recognize", methods=['GET','POST'])
# @login_required
# def recognize_faces():
#     form = RecognizeForm()
#     if form.validate_on_submit():
#         if form.images.data:
#             print("##############################################################\n")
#             print(form.images.data)
#             images = []
#             for image in form.images.data:
#                 recognized_image = faces.recognizeFaces(image)
#                 image_fn, image_path = save_image_to_user_images(form_image_name=image.filename, 
#                     to_be_saved_image=recognized_image)
#                 print(image_path)
#                 images.append(image_fn)
#         flash(f'Recognized faces have been marked.', 'success')
#         return render_template('recognized_faces.html', title="Recognized Faces", selectedListElement="recognizeFace", images=images)
#     return render_template('recognize_faces.html', title="Recognize Face", selectedListElement="recognizeFace", form=form)

@app.route("/faces/all")
@login_required
def all_faces():
    persons = Person.query.all()
    return render_template('all_faces.html', title='All Faces', selectedListElement="allFaces", persons=persons)

@app.route("/faces/<int:person_id>")
@login_required
def face(person_id):
    person = Person.query.get_or_404(person_id)
    datasets = Dataset.query.filter_by(author=person).all()
    print('###############################################################################')
    print(person)
    print(datasets)
    print('###############################################################################')
    return render_template('face.html', title=person.name, person=person, datasets=datasets)


@app.route("/faces/<int:person_id>/update", methods=['GET','POST'])
@login_required
def update_face(person_id):
    form = UpdateDatasetForm()
    person = Person.query.get_or_404(person_id)
    datasets = Dataset.query.filter_by(author=person).all()
    print('###############################################################################')
    print(person)
    print(datasets)
    print('###############################################################################')
    if form.validate_on_submit():
        if form.images.data:
            print("##############################################################\n")
            print(form.images.data)
            images = []
            # random_hex = secrets.token_hex(8)
            has_at_least_one_image_with_single_face = False
            for image in form.images.data:
                # TODO see if there is one only one face in the image (because suppose if there are 
                # 2 persons in the image and the 2nd one tries to recognize himself then if id folder
                # of 1st one comes first than the 2nd one's id folder, 2nd one will be recognized as
                # 1st person as the photo is in 1st person's id folder)
                
                # see if there is any image or not
                if image.mimetype.find("image") == -1:
                    break
                face_image = faces.hasSingleFace(image)
                if face_image is not None:
                    has_at_least_one_image_with_single_face = True
                    image_fn, image_path = save_image_to_dataset(dir_name=str(person_id), 
                        form_image_name=image.filename, to_be_saved_image=face_image)
                    dataset = Dataset(image_file=image_fn, author=person)
                    db.session.add(dataset)
                    print(image_path)
                    images.append(image_fn)
            name = form.name.data
            if name != person.name:
                person.name = name
            image_deleted_from_dataset = False
            if form.images_to_be_deleted.data:
                for image_id in form.images_to_be_deleted.data.split(";"):
                    dataset = Dataset.query.get(image_id)
                    path_to_image = './flaskapp/static/images/dataset/' + str(person_id) +'/' + dataset.image_file
                    if os.path.exists(path_to_image):
                        image_deleted_from_dataset = True
                        os.remove(path_to_image)
                        db.session.delete(dataset)
            db.session.commit()
            # update dataset_faces.dat if either an image was deleted from dataset or new image was added
            if has_at_least_one_image_with_single_face is True or image_deleted_from_dataset is True:
                faces.make_new_face_encodings()
            flash(f'Successfully updated {form.name.data}.', 'success')
            return redirect(url_for('face', person_id=person_id))
        else:
            flash(f'{form.name.data} not updated.', 'danger')
            return render_template('update_face.html', title=person.name, file_select=True, enableDeletePersonPhoto=True, datasets=datasets, form=form)

    elif request.method == 'GET':
        form.name.data = person.name
    return render_template('update_face.html', title=person.name, file_select=True, enableDeletePersonPhoto=True, datasets=datasets, form=form)


@app.route("/faces/<int:person_id>/delete", methods=['POST'])
@login_required
def delete_person(person_id):
    person = Person.query.get_or_404(person_id)
    db.session.delete(person)
    db.session.commit()
    shutil.rmtree('./flaskapp/static/images/dataset/' + str(person_id) +'/')
    faces.make_new_face_encodings()
    flash(f'Successfully deleted person - {person.name}!', 'success')
    return redirect(url_for('all_faces'))
    

# @app.route('/manifest.json')
# def manifest():
#     return send_from_directory('static', 'manifest.json')


@app.route('/sw.js')
def service_worker():
    responses = make_response(send_from_directory('static', 'sw.js'))
    responses.headers['Cache-Control'] = 'no-cache'
    return responses