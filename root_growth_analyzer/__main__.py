import logging
from os import environ
from sys import platform as sys_pf
from root_growth_analyzer.controller import AppController

if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")

def main():
    logging.basicConfig(level=environ.get('LOGLEVEL', 'INFO'))
    ctrl = AppController()
    ctrl.run()
