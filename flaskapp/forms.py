from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from flask_login import current_user
from wtforms import StringField, PasswordField, SubmitField, BooleanField, MultipleFileField, HiddenField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError
from flaskapp.models import User

class SignUpForm(FlaskForm):
    username = StringField('Username',
                           validators=[DataRequired(), Length(min=2, max=20)])
    email = StringField('Email',
                        validators=[DataRequired(), Email()])
    password = PasswordField('Password',
                             validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password',
                                     validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Sign Up')

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('This username is taken, please choose another. ')

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('This email is taken, please choose another. ')


class LoginForm(FlaskForm):
    email = StringField('Email',
                        validators=[DataRequired(), Email()])
    password = PasswordField('Password',
                             validators=[DataRequired()])
    remember = BooleanField('Remember Me')
    submit = SubmitField('Login')

class UpdateAccountForm(FlaskForm):
    username = StringField('Username',
                           validators=[DataRequired(), Length(min=2, max=20)])
    email = StringField('Email',
                        validators=[DataRequired(), Email()])
    picture = FileField('Update Profile Picture', validators=[FileAllowed(['jpg','png'])])
    submit = SubmitField('Update')

    def validate_username(self, username):
        if username.data != current_user.username:
            user = User.query.filter_by(username=username.data).first()
            if user:
                raise ValidationError('This username is taken, please choose another. ')

    def validate_email(self, email):
        if email.data != current_user.email:
            user = User.query.filter_by(email=email.data).first()
            if user:
                raise ValidationError('This email is taken, please choose another. ')

class RegistrationForm(FlaskForm):
	images = MultipleFileField('User Pictures', validators=[DataRequired(), FileAllowed(['jpg', 'png', 'jpeg'])])
	name = StringField('Name', validators=[DataRequired(), Length(min=2, max=40)])
	submit = SubmitField('Register')

# class RecognizeForm(FlaskForm):
# 	images = MultipleFileField('User Pictures', validators=[DataRequired(), FileAllowed(['jpg', 'png', 'jpeg'])])
# 	submit = SubmitField('Recognize')

class UpdateDatasetForm(FlaskForm):
	images = MultipleFileField('User Pictures', validators=[FileAllowed(['jpg', 'png', 'jpeg'])])
	name = StringField('New name', validators=[DataRequired(), Length(min=2, max=40)])
	images_to_be_deleted = HiddenField('Images to be deleted', validators=[])
	submit = SubmitField('Update') 