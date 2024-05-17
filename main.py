from flask import *

from database import *
from public import *


app=Flask(__name__)


app.secret_key='sparrow'
app.register_blueprint(public)




app.run(debug=True,port="5010") 