from pathlib import Path
import warnings

try:
    from pyasp.stdparse import StdFile, AnnotatedData, Header, StdMode
except ImportError:
    warnings.warn('Cannot find pyasp module. *.std output format is forbidden.')

    class StdFile:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError

    class AnnotatedData:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError

    class Header:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError

    class StdMode:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError

IISR_PATH = Path(__file__).parent.parent.resolve()
__version__ = '0.1'
