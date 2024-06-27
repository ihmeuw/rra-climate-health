from pathlib import Path
import sys
from rpy2.rinterface_lib import openrlib
# We're using an R library for modeling and R is so so dumb.  It doesn't know how to
# find its own libraries. So we have to tell it where to look.
openrlib.LD_LIBRARY_PATH = str(Path(sys.executable).parent.parent / 'lib')
del openrlib
del sys
del Path
