import unittest
from pysvso.lib.log import LoggerAdaptor, configure_logging
import logging
_logger = logging.getLogger("TestBase")

# please see https://stackoverflow.com/questions/8518043/turn-some-print-off-in-python-unittest
class MyTestResult(unittest.TextTestResult):

    def addSuccess(self, test):
        unittest.TestResult.addSuccess(self, test)
        self.stream.write("Successful with <%s.%s>!\n\n\r" % (test.__class__.__name__, test._testMethodName))

    def addError(self, test, err):
        unittest.TestResult.addError(self, test, err)
        self.stream.write("An Error Found!\n\n\r")

    def addFailure(self, test, err):
        unittest.TestResult.addFailure(self, test, err)
        self.stream.write("An Failure Found!\n\n\r")


# see http://python.net/crew/tbryan/UnitTestTalk/slide30.html
class MyTestRunner(unittest.TextTestRunner):
    def _makeResult(self):
        return MyTestResult(self.stream, self.descriptions, self.verbosity)

def runTest(*test_cases):
    suite = unittest.TestSuite()
    for test_case in test_cases:
        assert isinstance(test_case, unittest.TestCase)
        suite.addTest(test_case)
    runner = MyTestRunner()
    runner.run(suite)
    return runner

def Program():
    logging.basicConfig(level=logging.INFO, format=u"%(asctime)s [%(levelname)s]:%(filename)s, %(name)s, in line %(lineno)s >> %(message)s".encode('utf-8'))
    TEST_ALL = False
    if TEST_ALL:
        unittest.main(testRunner=MyTestRunner)
    else:
        # @todo : TODO
        # parsing test cases

        # run tests
        runTest()
    pass

if __name__ == "__main__":
    pass