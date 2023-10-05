"""
Seperated Cache module
- Avoid Circular imports
- Eable Cache in blueprints

Options:{SimpleCache,RedisCache,FileSystemCache}
Support multiple cache instances, each has differen backend.
It should be noted that SimpleCache(in memory) is neither thread or process safe.
Ref:
https://tamarisk.it/flask-caching-with-blueprints/
"""
from flask_caching import Cache

fc_config={
    'CACHE_TYPE': 'FileSystemCache',
    'CACHE_DIR': './cache',
    'CACHE_DEFAULT_TIMEOUT': 0,
}
sc_config={}
redis_config={}

cache = Cache(config=fc_config)
