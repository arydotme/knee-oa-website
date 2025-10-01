from flask import Flask, render_template
from flask_livereload import LiveReload

app = Flask(__name__)
livereload = LiveReload(app)

@app.route('/')
def hello_world():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)