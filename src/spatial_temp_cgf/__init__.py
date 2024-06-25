import os
import sys
from pathlib import Path
# We're using an R library for modeling and R is so so dumb.  It doesn't know how to
# find its own libraries. So we have to tell it where to look.
os.environ['LD_LIBRARY_PATH'] = (
    f"{os.environ['LD_LIBRARY_PATH']}:{Path(sys.executable).parent.parent / 'lib'}"
)

del os
del sys
del Path
