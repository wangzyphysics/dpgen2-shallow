import logging
import numpy as np
from pathlib import Path

from dpgen2.utils.dflow_query import(
    get_iteration,
    get_subkey,
)
from dflow import Workflow,download_artifact


class DownloadDefinition():
    def __init__(self):
        self.input_def = {}
        self.output_def = {}

    def add_def(
            self,
            tdict, 
            key,
            suffix = None,
    ):
        tdict[key] = suffix
        return self

    def add_input(
            self,
            input_key,
            suffix = None,
    ):
        return self.add_def(self.input_def, input_key, suffix)

    def add_output(
            self,
            output_key,
            suffix = None,
    ):
        return self.add_def(self.output_def, output_key, suffix)


op_download_setting = {
    "prep-run-train" : DownloadDefinition()\
    .add_input("init_models")\
    .add_input("init_data")\
    .add_input("iter_data")\
    .add_output("scripts")\
    .add_output("models")\
    .add_output("logs")\
    .add_output("lcurves"),
    "prep-run-lmp" : DownloadDefinition()\
    .add_output("logs")\
    .add_output("trajs")\
    .add_output("model_devis"),
    "prep-run-fp" : DownloadDefinition()\
    .add_input("confs")\
    .add_output("logs")\
    .add_output("labeled_data"),
    "collect-data" : DownloadDefinition()\
    .add_output("iter_data"),
}
    

def download_dpgen2_artifacts(
        wf,
        key,
        prefix = None,
):
    """
    download the artifacts of a step.
    the key should be of format 'iter-xxxxxx--subkey-of-step-xxxxxx'
    the input and output artifacts will be downloaded to 
    prefix/iter-xxxxxx/key-of-step/inputs/ and
    prefix/iter-xxxxxx/key-of-step/outputs/ 
    
    the downloaded input and output artifacts of steps are defined by
    `op_download_setting`

    """

    iteration = get_iteration(key)    
    subkey = get_subkey(key)
    mypath = Path(iteration)
    if prefix is not None:
        mypath = Path(prefix) / mypath
    
    dsetting = op_download_setting.get(subkey)
    if dsetting is None:
        logging.warning(f'cannot find download settings for {key}')
        return 

    input_def = dsetting.input_def
    output_def = dsetting.output_def

    step = wf.query_step(key=key)
    if len(step) == 0:
        raise RuntimeError(f'key {key} does not match any step')
    step = step[0]

    for kk in input_def.keys():
        pref = mypath / subkey / 'inputs'
        ksuff = input_def[kk]
        if ksuff is not None:
            pref = pref / ksuff
        try:
            download_artifact(
                step.inputs.artifacts[kk], 
                path=pref,
                skip_exists=True,
            )
        except (NotImplementedError, FileNotFoundError):
            # NotImplementedError to be compatible with old versions of dflow
            logging.warning(f'cannot download input artifact  {kk}  of  {key}, it may be empty')

    for kk in output_def.keys():
        pref = mypath / subkey / 'outputs'
        ksuff = output_def[kk]
        if ksuff is not None:
            pref = pref / ksuff
        try:
             download_artifact(
                step.outputs.artifacts[kk], 
                path=pref,
                skip_exists=True,
            )
        except (NotImplementedError, FileNotFoundError):
            # NotImplementedError to be compatible with old versions of dflow
            logging.warning(f'cannot download input artifact  {kk}  of  {key}, it may be empty')

    return
