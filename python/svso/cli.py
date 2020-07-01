import sys
import os
import argparse
import importlib

__author__ = "Lei Wang (yiak.wy@gmail.com)"
__date__ = "28-Aug-2018"
__update__ = "21-March-2019"
__license__ = "MIT"

ROOT = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, "%s/core"   % ROOT)
sys.path.insert(0, "%s/lib"    % ROOT)
sys.path.insert(0, "%s/cloud"  % ROOT)
sys.path.insert(0, "%s/cmds"   % ROOT)
sys.path.insert(0, "%s/config" % ROOT)
sys.path.insert(0, "%S/models" % ROOT)

# @todo : TODO
def ImportShell(shell_name):
    return None

def shell(raw_args):
    # you can implement shell selection logics here
    usage = """
cli.py [--<opt>]
    --subShell : enter into a sub shell program
    --prog: execute a program
    """

    parser = argparse.ArgumentParser(description=__doc__, usage=usage)
    parser.add_argument('-s', '--shell', help="enter into subShell routine")
    parser.add_argument('-e', '--exec', help="execute a program")
    parser.add_argument('argc', nargs='?', type=int)
    parser.add_argument('argv', nargs=argparse.REMAINDER, help="arguments for command")

    args = parser.parse_args(raw_args)

    if args.shell:
        subShell = ImportShell(args.shell)
    if subShell is not None:
        return subShell(raw_args)
    elif args.exec:
        [mod_path, method] = args.exec.rsplit('.', maxsplit=1)
        print("mod:", mod_path)
        print("method:", method)
        mod = importlib.import_module(mod_path)
        program = getattr(mod, method, None)
        if program is not None:
            return program(args.argv)
        else:
            print("The program <%s> is invalid!" % program)
    else:
        print("Not valid input!")
        parser.print_help()

if __name__ == "__main__":
    sys.exit(shell(sys.argv[1:]))
