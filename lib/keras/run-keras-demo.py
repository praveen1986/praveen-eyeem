import os
import time
import datetime
import logging
import flask
from flask.ext.basicauth import BasicAuth
import werkzeug
import tornado.wsgi
import tornado.httpserver
import urllib
import argparse
import yaml
import time
from termcolor import colored

from rnd_libs.lib.wrappers.concepts_wrapper import ConceptsWrapper
import keras.backend.tensorflow_backend as KTF
from keras import backend as K

from PIL import Image as PILImage
import cStringIO as StringIO

REPO_DIRNAME = os.path.abspath(os.path.dirname(__file__) + '/../..')
UPLOAD_FOLDER = '/tmp/keras_demos_uploads'
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg'])


def get_session(gpu_fraction):

    import tensorflow as tf

    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# Obtain the flask app object
app = flask.Flask(__name__)

app.config['BASIC_AUTH_USERNAME'] = 'deeplearning'
app.config['BASIC_AUTH_PASSWORD'] = 'parametersweep2016'

app.config['BASIC_AUTH_FORCE'] = True

basic_auth = BasicAuth(app)


@app.route('/secret')
@basic_auth.required
def secret_view():
    return render_template('secret.html')


@app.route('/')
def index():
    return flask.render_template('index.html', has_result=False, headline=app.headline)


@app.route('/classify_url', methods=['GET'])
def classify_url():

    imageurl = [flask.request.args.get('imageurl', '')]

    if imageurl[0].endswith('.gif') or imageurl[0].endswith('.GIF'):
        imageurl = app.clf.t.fetch_gif_frames(imageurl[0])

    image, _ = app.clf.t.fetch_images(imageurl)

    flag, predictions, confidence, timing = app.clf.t.classify_images(image)

    if len(predictions) > 1:
        predictions = app.clf.t.non_max(predictions)
        imagesrc = embed_image_html(imageurl[0])
    else:
        imagesrc = imageurl[0]

    result = []

    getty_like_filter = [app.clf.t.suppress_stop_list(g) for g in predictions]
    getty_like_safe = [app.clf.t.run_thresholds(g) for g in getty_like_filter]
    getty_like_pc = [app.clf.t.map_concepts(g) for g in getty_like_safe]
    getty_like_public = [app.clf.t.resolve_antonyms(g) for g in getty_like_pc]
    getty_like_unique = [app.clf.t.resolve_duplicates(g) for g in getty_like_public]

    result.append(predictions[0])
    result.append(getty_like_public[0])
    result.append(confidence[0])

    result.append(timing)

    app.clf.t.io.print_info(getty_like_unique[0])
    app.clf.t.io.print_info(confidence[0])

    return flask.render_template('index.html', has_result=True, result=result, imagesrc=imagesrc, headline=app.headline)


@app.route('/classify_upload', methods=['POST'])
def classify_upload():

    try:
        # We will save the file to disk for possible data collection.
        imagefile = flask.request.files['imagefile']
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + werkzeug.secure_filename(imagefile.filename)
        filename = os.path.join(UPLOAD_FOLDER, filename_)
        imagefile.save(filename)
        logging.info('Saving to %s.', filename)

        filename = [filename]

        if filename[0].endswith('.gif') or filename[0].endswith('.GIF'):
            filename = app.clf.fetch_gif_frames(filename[0])

        image, _ = app.clf.t.fetch_images(filename)

    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        return flask.render_template('index.html', has_result=True, result=(False, 'Cannot open uploaded image.'))

    flag, predictions, confidence, timing = app.clf.t.classify_images(image)

    if len(predictions) > 1:
        predictions = app.clf.t.non_max(predictions)

    result = []

    getty_like_filter = [app.clf.t.suppress_stop_list(g) for g in predictions]
    getty_like_safe = [app.clf.t.run_thresholds(g) for g in getty_like_filter]
    getty_like_pc = [app.clf.t.map_concepts(g) for g in getty_like_safe]
    getty_like_public = [app.clf.t.resolve_antonyms(g) for g in getty_like_pc]
    getty_like_unique = [app.clf.t.resolve_duplicates(g) for g in getty_like_public]

    result.append(predictions[0])
    result.append(getty_like_public[0])
    result.append(confidence[0])

    result.append(timing)

    app.clf.t.io.print_info(getty_like_unique[0])
    app.clf.t.io.print_info(confidence[0])

    return flask.render_template(
        'index.html', has_result=True, result=result,
        imagesrc=embed_image_html(filename[0]),
        headline=app.headline
    )


def embed_image_html(image_path):
        """Creates an image embedded in HTML base64 format."""
        image_pil = PILImage.open(image_path)
        image_pil = image_pil.resize((512, 512))
        string_buf = StringIO.StringIO()
        image_pil.save(string_buf, format='png')
        data = string_buf.getvalue().encode('base64').replace('\n', '')
        return 'data:image/png;base64,' + data


def allowed_file(filename):
        return ('.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS)


def start_tornado(app, port=5000):

    http_server = tornado.httpserver.HTTPServer(tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


def start_from_terminal(app, opts):

    app.clf = ConceptsWrapper('eyeem-uploads', opts.config_file)
    app.headline = opts.headline

    if not app.clf.t.init:
        print colored('[ERR] Could not initialize network', 'red')
        return

    if opts.debug:
        app.run(debug=True, host='0.0.0.0', port=opts.port)
    else:
        start_tornado(app, opts.port)

if __name__ == '__main__':

    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description='Utility for keras driven multi-label classification demo')
    parser.add_argument('-d', '--debug', help='Enable debug mode for the utility', action="store_true", default=False)
    parser.add_argument('-p', '--port', help='which port to run the keras demo on', type=int, default=1000)
    parser.add_argument('-k', '--config-file', help="Config file to set-up Network", dest="config_file", required=True)
    parser.add_argument('--gpu-limit', dest='gpu_limit', type=float, default=0.5, help='Use fraction of GPU memory')
    parser.add_argument('--headline', dest='headline', type=str, default='Image Tagging', help='Headline to be used for the demo')

    args = parser.parse_args()

    if K._BACKEND == 'tensorflow':
        KTF.set_session(get_session(args.gpu_limit))

    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    print args.headline

    start_from_terminal(app, args)
