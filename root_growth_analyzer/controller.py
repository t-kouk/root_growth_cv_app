import argparse
import logging
from os import environ

from root_growth_analyzer.root_growth_gui import (RootGrowthGUI, pipeline_run)

DEFAULTS = {
    'INPUT': 'images/',
    'OUTPUT_DATA': 'images--processed/',
    'TIP_SIZE': '10',
    'SCENARIO': 2
}

class AppController():
    def __init__(self):
        self.arg_parser = argparse.ArgumentParser()
        default_in = DEFAULTS['INPUT']
        default_out_img = DEFAULTS['OUTPUT_DATA']
        default_root_tip_size = DEFAULTS['TIP_SIZE']
        default_scenario = DEFAULTS['SCENARIO']
        self.arg_parser.add_argument(
            '--cli',
            dest='use_cli',
            action='store_true',
            help=f'Use --cli flag to run the program from the command line.'
        )
        self.arg_parser.add_argument(
            '--input',
            dest='input',
            default=default_in,
            type=str,
            help=f'Set path for input image directory. Default: {default_in}'
        )
        self.arg_parser.add_argument(
            '--output',
            dest='outimg',
            default=default_out_img,
            type=str,
            help=f'Set directory for processed images. Default: {default_out_img}'
        )
        self.arg_parser.add_argument(
            '--tip-size',
            dest='tip_size',
            default=default_root_tip_size,
            type=str,
            help=f'Set minimum root tip size searched (mm). Default: {default_root_tip_size}'
        )
        self.arg_parser.add_argument(
            '--scenario',
            dest='scenario',
            default=default_scenario,
            type=int,
            help=f'Set 1 for single image analysis or 2 for analyzing root growth from sequentially paired images. Default: {default_scenario}'
        )

    def run(self):
        args = self.arg_parser.parse_args()
        if args.use_cli:
            if '../' in args.input or '../' in args.outimg:
                logging.error('Relative paths to parent directories is not supported, please use full paths instead')
                return
            if args.input != DEFAULTS['INPUT'] and args.outimg == DEFAULTS['OUTPUT_DATA']:
                outimg = f'{args.input.split("/")[-1]}--processed'
            else:
                outimg = args.outimg
            pipeline_run(args.input, outimg, args.tip_size, args.scenario)
        else:
            RootGrowthGUI(DEFAULTS)
