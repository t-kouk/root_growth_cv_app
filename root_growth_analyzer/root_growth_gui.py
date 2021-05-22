import os
import re
import logging
from time import sleep
from threading import (
    Thread,
    Event
)
from guizero import (
    App,
    Box,
    Text,
    TextBox,
    PushButton,
    Combo,
    select_folder
)
from tkinter import ttk

from root_growth_analyzer.DL_model.predict_imgs import DLModel
from root_growth_analyzer.root_tips.scenario1 import single_image_analysis
from root_growth_analyzer.root_tips.scenario2 import root_tip_analysis


def pipeline_run(input_path, output_folder, root_tip_size, scenario, progress=None, exit_flag=None):
    """
    Make image segmentation and compute root tip growth, save results.
    """
    if scenario == 1:
        logging.info("Starting process for single image analysis")
    elif scenario == 2:
        logging.info("Starting process for root growth analysis")
    else:
        logging.error("Scenario should be either 1 or 2. Terminating.")
        return
    img_names = sorted(os.listdir(input_path))
    step = 50/len(img_names)
    progress['value'] = 0.5
    if not os.path.exists(input_path):
        raise ValueError("Input path is invalid! Directory {} does not exist".format(input_path))
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    dl_images = f'{output_folder}/DL_images'
    if not os.path.exists(dl_images):
        os.mkdir(dl_images)
    if os.listdir(dl_images):
        logging.warning('Folder for processed images is not empty and any duplicate images will be overwritten. Use CTRL-C to abort.')
        sleep(5)

    DLModel().apply_dl_model(input_path, dl_images, progress, step, exit_flag)
    if exit_flag and exit_flag.is_set():
        return

    root_tip_dir = f'{output_folder}/root_tip_analysis/'
    if not os.path.exists(root_tip_dir):
        os.mkdir(root_tip_dir)
    progress['value'] = 50

    if scenario == 1:
        for img_name in img_names:
            if exit_flag and exit_flag.is_set():
                logging.info("Received exit signal, terminating run")
                return
            img_path = f'{input_path}/{img_name}'

            img_pred_name = f'{img_name.strip(".jpg")}-prediction.jpg'
            img_pred_path = f'{dl_images}/{img_pred_name}'

            date = re.search('20[0-9][0-9]\.[0-1][0-9]\.[0-3][0-9]', img_name).group(0)
            single_image_analysis(
                img_pred_path,
                img_path,
                root_tip_size,
                root_tip_dir,
                date
            )
            progress.step(step)

    elif scenario == 2:
        aggr_results = 'Date;Avg difference, mm\n'
        prev_pred_path = None
        for img_name in img_names:
            if exit_flag and exit_flag.is_set():
                logging.info("Received exit signal, terminating run")
                return
            img_path = f'{input_path}/{img_name}'

            img_pred_name = f'{img_name.strip(".jpg")}-prediction.jpg'
            img_pred_path = f'{dl_images}/{img_pred_name}'

            if prev_pred_path:
                date = re.search('20[0-9][0-9]\.[0-1][0-9]\.[0-3][0-9]', img_name).group(0)
                df = root_tip_analysis(
                    prev_pred_path,
                    img_pred_path,
                    img_path,
                    root_tip_size,
                    root_tip_dir,
                    date
                )
                av = df['Difference, mm'].mean() if df is not None else 0
                aggr_results += f'{date};{av}\n'

            prev_pred_path = img_pred_path
            progress.step(step)

        aggr_results_path = f'{output_folder}/avg_results.csv'
        with open(aggr_results_path, 'w') as out:
            logging.info(f"Writing average results to {aggr_results_path}")
            out.write(aggr_results)

    progress['value'] = 100
    progress.stop()
    progress.grid_remove()

class RootGrowthGUI():
    thread = None
    output_user_set = False
    exit_flag = Event()

    def __init__(self, defaults):
        self.app = App('Root image analyzer', width=700, height=300, layout='grid')

        self.cwd = os.getcwd()
        self.input_path = defaults['INPUT']
        self.output_dir_path = f'{defaults["INPUT"].strip("/").split("/")[-1]}--processed/'

        Text(self.app, text='Select image folder', grid=[0,0], align='left')
        input_button = PushButton(self.app, command=self.choose_input, text='Choose', grid=[1,0], align='left')
        self.input_path_text = Text(self.app, text=self.input_path, grid=[2,0])

        Text(self.app, text='Set minimum root tip size searched (mm)', grid=[0,1], align='left')
        self.tip_size = TextBox(self.app, text=defaults['TIP_SIZE'], grid=[1,1], align='left')

        Text(self.app, text='Choose directory for processed images and data', grid=[0,3], align='left')
        output_dir_button = PushButton(self.app, command=self.choose_out_dir, text='Choose', grid=[1,3], align='left')
        self.output_dir_text = Text(self.app, text=self.output_dir_path, grid=[2,3])

        self.scenario_descs = {
            'Single images analysis': 1,
            'Sequential root growth analysis': 2
        }
        self.scenario_select = Combo(
            self.app,
            options=self.scenario_descs.keys(),
            selected='Sequential root growth analysis',
            grid=[0,4],
            align='left'
        )

        start_button = PushButton(self.app, command=self.start_model, text='Start', grid=[1,5], align='left')
        exit_button = PushButton(self.app, command=self.exit, text='Exit', grid=[2,5], align='left')

        self.app.display()


    def choose_input(self):
        self.input_path = select_folder(title='Select folder', folder=self.cwd)
        self.input_path_text = Text(self.app, text=self.input_path, grid=[2,0])
        if not self.output_user_set:
            self.output_dir_path = f'{self.input_path.split("/")[-1]}--processed'
            self.output_dir_text = Text(self.app, text=self.output_dir_path, grid=[2,3])


    def choose_out_dir(self):
        self.output_dir_path = select_folder(title='Select folder', folder=self.cwd)
        self.output_dir_text = Text(self.app, text=self.output_dir_path, grid=[2,3])
        self.output_user_set = True


    def start_model(self):
        if self.thread and self.thread.is_alive():
            return
        size = self.tip_size.value
        scenario = self.scenario_descs[self.scenario_select.value]
        progress_bar = ttk.Progressbar(self.app.tk, orient='horizontal', length=260)
        progress_bar.grid(column=0, row=6, columnspan=4, ipady=10)
        progress_bar.config(mode='determinate', maximum=100, value=0)
        self.thread = Thread(
            target=pipeline_run,
            args=(self.input_path, self.output_dir_path, size, scenario, progress_bar, self.exit_flag)
        )
        self.thread.start()


    def exit(self):
        self.exit_flag.set()
        self.app.destroy()
