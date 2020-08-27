import unittest
from pysvso.lib.log import LoggerAdaptor, configure_logging
import logging
_logger = logging.getLogger("TestBase")

from pysvso.lib.test import runTest

# @todo : TODO
# writing unit tests for CI/CD of exporting models to CPP inference engine

# @todo ; TODO
class SavedModelFormatTestCase(unittest.TestCase):

    logger = LoggerAdaptor("tests/models/test_sfe.SavedModelFormatTestCase", _logger)

    @classmethod
    def setUpClass(cls):
        pass

    pass

# @todo : TODO
class StaticGraphFormatTestCase(unittest.TestCase):

    logger = LoggerAdaptor("tests/models/test_sfe.StaticGraphFormatTestCase", _logger)

    @classmethod
    def setUpClass(cls):
        pass

    pass

# @todo : TODO(change base model)