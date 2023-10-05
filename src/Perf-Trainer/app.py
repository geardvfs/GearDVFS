import flask
from flask_caching import Cache

# cache module
from util_modules.cache import cache
# import blueprints
from util_modules.test_module import test_api, basic_api
from util_modules.dqn_module import nn_api
from util_modules.context_module import context_api
app = flask.Flask(__name__)
cache.init_app(app)

app.register_blueprint(test_api)
app.register_blueprint(basic_api)
app.register_blueprint(nn_api)
app.register_blueprint(context_api)

if __name__ == "__main__":
    
    app.run(host='192.168.137.1', port=5000, debug=True,threaded=True)

 