from flask import Flask, render_template, request, make_response, send_file

import os
import pandas as pd
from testing_called_from_cloud import CloudTest

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    x = request.files['x']
    df = pd.read_csv(x)
    cloudtest_instance = CloudTest(df)
    prediction = cloudtest_instance.main()

    # Create a response object with the output data as a CSV attachment
    output_csv = pd.DataFrame({'output': prediction})
    output_csv['output'] = output_csv['output'].astype(int)
    csv = output_csv.to_string(index=False)

    # Set the output filename
    output_filename = 'output.csv'

    # Save the output CSV file to disk
    with open(output_filename, 'w') as f:
        f.write(csv)

    # Render the template with the output data and filename

    return render_template('index.html', prediction_text=" Prediction was successful! Here are the first 15 values:       {}".format(prediction.head(15).astype(int).to_string(index=False)),output_csv=csv, output_filename=output_filename)

@app.route('/download', methods=['GET'])
def download():
    output_filename = 'output.csv'
    mimetype = 'text/csv'
    return send_file(output_filename, mimetype=mimetype, as_attachment=True, download_name='output.csv')


if __name__ == '__main__':
    app.run(debug=True)