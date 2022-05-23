from flask import Flask, render_template, request, redirect, abort
from models import db, UserModel
import pymysql
pymysql.install_as_MySQLdb()

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:root@db/docker_practice_v3'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)


@app.before_first_request
def create_table():
    db.create_all()


@app.route('/')
def start():
    allUsers = RetrieveUserList()
    return render_template('home.html', users=allUsers)




def RetrieveUserList():
    users = UserModel.query.all()
    return users





if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
