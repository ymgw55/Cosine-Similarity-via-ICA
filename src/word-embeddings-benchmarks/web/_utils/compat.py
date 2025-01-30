"""
Compatibility layer for Python 3/Python 2 single codebase
"""
import hashlib
import sys

if sys.version_info[0] == 3:
    import io
    import pickle
    # import urllib
    import urllib.request

    _basestring = str
    cPickle = pickle
    StringIO = io.StringIO
    BytesIO = io.BytesIO
    _urllib = urllib
    izip = zip

    def md5_hash(string):
        m = hashlib.md5()
        m.update(string.encode('utf-8'))
        return m.hexdigest()
else:
    import itertools
    import types
    # import urllib
    import urllib.request

    import cPickle
    import StringIO
    import urllib2
    import urlparse

    _basestring = basestring
    cPickle = cPickle
    StringIO = BytesIO = StringIO.StringIO
    izip = itertools.izip

    class _module_lookup(object):
        modules = [urlparse, urllib2, urllib]

        def __getattr__(self, name):
            for module in self.modules:
                if hasattr(module, name):
                    attr = getattr(module, name)
                    if not isinstance(attr, types.ModuleType):
                        return attr
            raise NotImplementedError(
                'This function has not been imported properly')

    module_lookup = _module_lookup()

    class _urllib():
        request = module_lookup
        error = module_lookup
        parse = module_lookup

    def md5_hash(string):
        m = hashlib.md5()
        m.update(string)
        return m.hexdigest()
