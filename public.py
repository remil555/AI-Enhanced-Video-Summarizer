import uuid
from database import *
from flask import *
from cap import *



public=Blueprint('public',__name__)


@public.route('/')
def home():
    return render_template('home.html')

@public.route('/input', methods=['POST', 'GET'])
def input():
    escape_generated_text=""
    if request.method == 'POST' and 'video' in request.files:
        video = request.files['video']
        if video.filename != '':
            try:
                # Save the uploaded video file to a static directory
                path = "static/" + str(uuid.uuid4()) + video.filename
                video.save(path)
                
                # Generate summary for the uploaded video
                text=generate_summary(path)
                print("1"*100,"\nTEXT : ",text)
                escape_generated_text=text.replace("'","''")
                print(escape_generated_text,'??//////////////')
                

                
                # Optionally, you can return the summary to the client
                return render_template('input.html',escape_generated_text=escape_generated_text,path=path,status='success')
                
               
            except Exception as e:
                # Handle any errors that occur during file saving or summary generation
                return '''<script>alert("error");window.location="/input"</script>'''
        else:
            return '''<script>alert("upload file");window.location="/input"</script>'''
            
    
    # Render the input form template for GET requests or when 'submit' is not in request.form
    return render_template('input.html')
