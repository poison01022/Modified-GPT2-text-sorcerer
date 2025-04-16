from flask import Flask, request, render_template
from generate import generate_text

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        prompt = request.form["prompt"]
        result = generate_text(prompt)
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
