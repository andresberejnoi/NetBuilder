#from . import examples
#from . import tests
from . import __version__
from .__version__ import __version__
from . import activations
from .activations import *
from . import loss
from .loss import *
from . import debug_test
from .debug_test import *
from . import file_operations
from .file_operations import *
from . import _param_keys
#import _param_keys as keys

from . import NeuralNet
from .NeuralNet import Network,NetworkError #import this at the end
