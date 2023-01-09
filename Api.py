from flask import Flask,request
from flask_restful import Resource,Api,reqparse
from Bot import Chatbot

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument("statement", help="statement is required")

class BotApi(Resource):
    def get(self):
        return {'Hello':"Xen"}
    
    def post(self):
        arg = parser.parse_args()['statement']
        response = Chatbot(arg)
        return response
        

api.add_resource(BotApi,'/')

if __name__ == "__main__":
    app.run(debug=True,port=4433)