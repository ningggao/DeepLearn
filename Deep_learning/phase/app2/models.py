from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class UserModel(db.Model):
    __tablename__ = "usertable"

    id = db.Column(db.Integer, primary_key=True)
    UserID = db.Column(db.String(20))
    name = db.Column(db.String(20))

    def __init__(self, UserID, name):
        self.UserID = UserID
        self.name = name

    def __repr__(self):
        return f"{self.UserID} {self.name}"
