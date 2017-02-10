import os
import time
import datetime
import logging
import flask
import werkzeug
import optparse
import tornado.wsgi
import tornado.httpserver
import urllib
import argparse
import yaml
import pdb
import numpy as np
from PIL import Image as PILImage
import cStringIO as StringIO
from eyelibs.lib.full_conv_net.all_conv import FullyConvNetwork
from eyelibs.lib.label_embedding.label_embedding import LabelEmbedding

REPO_DIRNAME = os.path.abspath(os.path.dirname(__file__) + '/../..')
UPLOAD_FOLDER = '/tmp/caffe_demos_uploads'
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg'])

# Obtain the flask app object
app = flask.Flask(__name__)

@app.route('/')
def index():
    return flask.render_template('index.html', has_result=False)


@app.route('/classify_url', methods=['GET'])
def classify_url():

    imageurl = flask.request.args.get('imageurl', '')
    color_image, image = app.clf.fetch_image(imageurl)

    image,mask_image = app.clf.check_for_size(image)
    flag, dl_pred, hedge_pred, entry_pred, timing, dense_pred,_,_ = app.clf.classify_image(image,mask_image)

    dl_pred, hedge_pred, entry_pred = [l[0] for l in dl_pred], [l[0] for l in hedge_pred], [l[0] for l in entry_pred]

    result = [ flag, dl_pred, hedge_pred, entry_pred]
    _label_emb_pred = app.emb.predict(dense_pred)
    label_emb_pred = app.emb.resolve_antonyms(_label_emb_pred)

    concepts = [ d[0] for d in  label_emb_pred ]
    scores = [ d[1] for d in  label_emb_pred ]

    concepts = app.emb.map_lemb_concepts(concepts)

    result.append([ (g,s) for g,s in zip(concepts,scores)])
    result.append(timing)

    return flask.render_template(
        'index.html', has_result=True, result=result, imagesrc=imageurl)


@app.route('/classify_upload', methods=['POST'])
def classify_upload():
    try:
        # We will save the file to disk for possible data collection.
        imagefile = flask.request.files['imagefile']
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
            werkzeug.secure_filename(imagefile.filename)
        filename = os.path.join(UPLOAD_FOLDER, filename_)
        imagefile.save(filename)
        logging.info('Saving to %s.', filename)

        color_image,image = app.clf.fetch_image(filename)

        n_image,mask_image = app.clf.check_for_size(image)

    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open uploaded image.')
        )

    flag, dl_pred, hedge_pred, entry_pred, timing, dense_pred, _, _ = app.clf.classify_image(n_image,mask_image)
    dl_pred, hedge_pred, entry_pred = [l[0] for l in dl_pred], [l[0] for l in hedge_pred], [l[0] for l in entry_pred]

    result = [ flag, dl_pred, hedge_pred, entry_pred]
    _label_emb_pred = app.emb.predict(dense_pred)
    label_emb_pred = app.emb.resolve_antonyms(_label_emb_pred)

    concepts = [ d[0] for d in  label_emb_pred ]
    scores = [ d[1] for d in  label_emb_pred ]

    concepts = app.emb.map_lemb_concepts(concepts)

    result.append([ (g,s) for g,s in zip(concepts,scores)])
    result.append(timing)

    return flask.render_template(
        'index.html', has_result=True, result=result,
        imagesrc=embed_image_html(image)
    )


def embed_image_html(image):
    """Creates an image embedded in HTML base64 format."""
    image_pil = PILImage.fromarray((255 * image).astype('uint8'))
    image_pil = image_pil.resize((512, 512))
    string_buf = StringIO.StringIO()
    image_pil.save(string_buf, format='png')
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    return 'data:image/png;base64,' + data


def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS
    )

def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


def start_from_terminal(app,opts):

    #
    # reading yaml config file
    #
    fp = open(opts.config_file,'r')
    dl_config = yaml.load(fp)
    fp.close()
    #

    app.clf = FullyConvNetwork(dl_config,opts.config_file)
    app.clf.init_network()

    app.emb = LabelEmbedding(opts.emb_config_file,False)
    app.emb.prepare_for_testing()

    if not app.clf.init_flag:
        print colored('[ERR] Could not initialize network','red')
        return
    #if

    if opts.debug:
        app.run(debug=True, host='0.0.0.0', port=opts.port)
    else:
        start_tornado(app, opts.port)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description='Utility for fully-convolution-NN demo')
    parser.add_argument('-d','--debug',help='Enable debug mode for the utility',
                        action="store_true",default=False)
    parser.add_argument('-p','--port',help='which port to run the hedging demo on',
                        type=int,default=1000)
    parser.add_argument('-g','--use-gpu',
                        help="use gpu mode",dest='gpu',
                        action='store_true', default=False)
    parser.add_argument('-k','--config-file',
                        help="Config file to set-up classifier",
                        dest="config_file",required=True)
    parser.add_argument('-x','--label-em-config-file',
                        help="Config file to set-up Label embedding",
                        dest="emb_config_file",required=True)
    parser.add_argument('-s','--server-name',
                        help="Server Name",
                        dest="server_name",default='skynet.rad.eyeem.com')
    parser.add_argument('-f','--filter-with-thresholds',
                        help="Keep only non-suppressed identified concepts",
                        dest="filter_with_threshold",action='store_true',default=False)
    args = parser.parse_args()
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    start_from_terminal(app,args)
