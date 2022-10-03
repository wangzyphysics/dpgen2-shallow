import glob, dpdata, os, pickle
from pathlib import Path
from dflow import (
    InputParameter,
    OutputParameter,
    Inputs,
    InputArtifact,
    Outputs,
    OutputArtifact,
    Workflow,
    Step,
    Steps,
    upload_artifact,
    download_artifact,
    S3Artifact,
    argo_range,
)
from dflow.python import (
    PythonOPTemplate,
    OP,
    OPIO,
    OPIOSign,
    Artifact,
    upload_packages,
    FatalError,
    TransientError,
)

from dpgen2.op import (
    PrepDPTrain,
    RunDPTrain,
    PrepLmp,
    RunLmp,
    PrepVasp,
    RunVasp,
    SelectConfs,
    CollectData,
)
from dpgen2.superop import (
    PrepRunDPTrain,
    PrepRunLmp,
    PrepRunFp,
    ConcurrentLearningBlock,
)
from dpgen2.flow import (
    ConcurrentLearning,
)
from dpgen2.fp import (
    VaspInputs,
)
from dpgen2.exploration.scheduler import (
    ExplorationScheduler,
    ConvergenceCheckStageScheduler,
)
from dpgen2.exploration.task import (
    ExplorationStage,
    ExplorationTask,
    NPTTaskGroup,
)
from dpgen2.exploration.selector import (
    ConfSelectorLammpsFrames,
    TrustLevel,
)
from dpgen2.constants import (
    default_image,
    default_host,
)
from dpgen2.utils import (
    dump_object_to_file,
    load_object_from_file,
    normalize_alloy_conf_dict,
    generate_alloy_conf_file_content,
    sort_slice_ops,
    print_keys_in_nice_format,
    workflow_config_from_dict,
)
from dpgen2.utils.step_config import normalize as normalize_step_dict
from dpgen2.entrypoint.submit_args import normalize as normalize_submit_args
from typing import (
    Union, List, Dict, Optional,
)

default_config = normalize_step_dict(
    {
        "template_config" : {
            "image" : default_image,
        }
    }
)

def expand_sys_str(root_dir: Union[str, Path]) -> List[str]:
    root_dir = Path(root_dir)
    matches = [str(d) for d in root_dir.rglob("*") if (d / "type.raw").is_file()]
    if (root_dir / "type.raw").is_file():
        matches.append(str(root_dir))
    return matches

def make_concurrent_learning_op (
        train_style : str = 'dp',
        explore_style : str = 'lmp',
        fp_style : str = 'vasp',
        prep_train_config : str = default_config,
        run_train_config : str = default_config,
        prep_explore_config : str = default_config,
        run_explore_config : str = default_config,
        prep_fp_config : str = default_config,
        run_fp_config : str = default_config,
        select_confs_config : str = default_config,
        collect_data_config : str = default_config,
        cl_step_config : str = default_config,
        upload_python_package : bool = None,
):
    if train_style == 'dp':
        prep_run_train_op = PrepRunDPTrain(
            "prep-run-dp-train",
            PrepDPTrain,
            RunDPTrain,
            prep_config = prep_train_config,
            run_config = run_train_config,
            upload_python_package = upload_python_package,
        )
    else:
        raise RuntimeError(f'unknown train_style {train_style}')
    if explore_style == 'lmp':
        prep_run_explore_op = PrepRunLmp(
            "prep-run-lmp",
            PrepLmp,
            RunLmp,
            prep_config = prep_explore_config,
            run_config = run_explore_config,
            upload_python_package = upload_python_package,
        )
    else:
        raise RuntimeError(f'unknown explore_style {explore_style}')
    if fp_style == 'vasp':
        prep_run_fp_op = PrepRunFp(
            "prep-run-vasp",
            PrepVasp,
            RunVasp,
            prep_config = prep_fp_config,
            run_config = run_fp_config,
            upload_python_package = upload_python_package,
        )
    else:
        raise RuntimeError(f'unknown fp_style {fp_style}')

    # ConcurrentLearningBlock
    block_cl_op = ConcurrentLearningBlock(
        "concurrent-learning-block", 
        prep_run_train_op,
        prep_run_explore_op,
        SelectConfs,
        prep_run_fp_op,
        CollectData,
        select_confs_config = select_confs_config,
        collect_data_config = collect_data_config,
        upload_python_package = upload_python_package,
    )    
    # dpgen
    dpgen_op = ConcurrentLearning(
        "concurrent-learning",
        block_cl_op,
        upload_python_package = upload_python_package,
        step_config = cl_step_config,
    )
        
    return dpgen_op


def make_conf_list(
        conf_list,
        type_map,
        fmt='vasp/poscar'
):
    # load confs from files
    if isinstance(conf_list, list):
        conf_list_fname = []
        for jj in conf_list:
            confs = sorted(glob.glob(jj))
            conf_list_fname = conf_list_fname + confs
        conf_list = []
        for ii in conf_list_fname:
            ss = dpdata.System(ii, type_map=type_map, fmt=fmt)
            ss.to('lammps/lmp', 'tmp.lmp')
            conf_list.append(Path('tmp.lmp').read_text())
        if Path('tmp.lmp').is_file():
            os.remove('tmp.lmp')
    # generate alloy confs
    elif isinstance(conf_list, dict):
        conf_list['type_map'] = type_map
        i_dict = normalize_alloy_conf_dict(conf_list)
        conf_list = generate_alloy_conf_file_content(**i_dict)
    else:
        raise RuntimeError('unknown input format of conf_list: ', type(conf_list))
    return conf_list


def make_naive_exploration_scheduler(
        config,
        old_style = False,
):
    # use npt task group
    model_devi_jobs = config['model_devi_jobs'] if old_style else config['explore']['stages']
    sys_configs = config['sys_configs'] if old_style else config['explore']['configurations']
    sys_prefix = config.get('sys_prefix')
    if sys_prefix is not None:
        for ii in range(len(sys_configs)):
            if isinstance(sys_configs[ii], list):
                sys_configs[ii] = [os.path.join(sys_prefix, jj) for jj in sys_prefix[ii]]
    mass_map = config['mass_map'] if old_style else config['inputs']['mass_map']
    type_map = config['type_map'] if old_style else config['inputs']['type_map']
    numb_models = config['numb_models'] if old_style else config['train']['numb_models']
    fp_task_max = config['fp_task_max'] if old_style else config['fp']['task_max']
    conv_accuracy = config['conv_accuracy'] if old_style else config['explore']['conv_accuracy']
    max_numb_iter = config['max_numb_iter'] if old_style else config['explore']['max_numb_iter']
    fatal_at_max = config.get('fatal_at_max', True) if old_style else config['explore']['fatal_at_max']
    scheduler = ExplorationScheduler()

    for job in model_devi_jobs:
        # task group
        tgroup = NPTTaskGroup()
        ##  ignore the expansion of sys_idx
        # get all file names of md initial configurations
        try:
            sys_idx = job['sys_idx']
        except KeyError:
            sys_idx = job['conf_idx']
        conf_list = []        
        for ii in sys_idx:
            conf_list += make_conf_list(sys_configs[ii], type_map)
        # add the list to task group
        n_sample = job.get('n_sample')
        tgroup.set_conf(
            conf_list,
            n_sample=n_sample,
        )
        temps = job['temps']
        press = job['press']
        trj_freq = job['trj_freq']
        nsteps = job['nsteps']
        ensemble = job['ensemble']
        # add md settings
        tgroup.set_md(
            numb_models,
            mass_map,
            temps = temps,
            press = press,
            ens = ensemble,
            nsteps = nsteps,
        )
        tasks = tgroup.make_task()
        # stage
        stage = ExplorationStage()
        stage.add_task_group(tasks)
        # trust level
        trust_level = TrustLevel(
            config['model_devi_f_trust_lo'] if old_style else config['explore']['f_trust_lo'],
            config['model_devi_f_trust_hi'] if old_style else config['explore']['f_trust_hi'],
            level_v_lo=config.get('model_devi_v_trust_lo') if old_style else config['explore']['v_trust_lo'],
            level_v_hi=config.get('model_devi_v_trust_hi') if old_style else config['explore']['v_trust_hi'],
        )
        # selector
        selector = ConfSelectorLammpsFrames(
            trust_level,
            fp_task_max,
        )
        # stage_scheduler
        stage_scheduler = ConvergenceCheckStageScheduler(
            stage,
            selector,
            conv_accuracy = conv_accuracy,
            max_numb_iter = max_numb_iter,
            fatal_at_max = fatal_at_max,
        )
        # scheduler
        scheduler.add_stage_scheduler(stage_scheduler)
        
    return scheduler


def get_kspacing_kgamma_from_incar(
        fname,
):
    with open(fname) as fp:
        lines = fp.readlines()
    for ii in lines:
        if 'KSPACING' in ii:
            ks = float(ii.split('=')[1])
        if 'KGAMMA' in ii:
            if 'T' in ii.split('=')[1]:
                kg = True
            elif 'F' in ii.split('=')[1]:
                kg = False
            else:
                raise RuntimeError(f"invalid kgamma value {ii.split('=')[1]}")
    return ks, kg


def workflow_concurrent_learning(
        config : Dict,
        old_style : Optional[bool] = False,
):
    default_config = normalize_step_dict(config.get('default_config', {})) if old_style else config['default_step_config']

    train_style = config.get('train_style', 'dp') if old_style else config['train']['type']
    explore_style = config.get('explore_style', 'lmp') if old_style else config['explore']['type']
    fp_style = config.get('fp_style', 'vasp') if old_style else config['fp']['type']
    prep_train_config = normalize_step_dict(config.get('prep_train_config', default_config)) if old_style else config['step_configs']['prep_train_config']
    run_train_config = normalize_step_dict(config.get('run_train_config', default_config)) if old_style else config['step_configs']['run_train_config']
    prep_explore_config = normalize_step_dict(config.get('prep_explore_config', default_config)) if old_style else config['step_configs']['prep_explore_config']
    run_explore_config = normalize_step_dict(config.get('run_explore_config', default_config)) if old_style else config['step_configs']['run_explore_config']
    prep_fp_config = normalize_step_dict(config.get('prep_fp_config', default_config)) if old_style else config['step_configs']['prep_fp_config']
    run_fp_config = normalize_step_dict(config.get('run_fp_config', default_config)) if old_style else config['step_configs']['run_fp_config']
    select_confs_config = normalize_step_dict(config.get('select_confs_config', default_config)) if old_style else config['step_configs']['select_confs_config']
    collect_data_config = normalize_step_dict(config.get('collect_data_config', default_config)) if old_style else config['step_configs']['collect_data_config']
    cl_step_config = normalize_step_dict(config.get('cl_step_config', default_config)) if old_style else config['step_configs']['cl_step_config']
    upload_python_package = config.get('upload_python_package', None)
    init_models_paths = config.get('training_iter0_model_path')

    concurrent_learning_op = make_concurrent_learning_op(
        train_style,
        explore_style,
        fp_style,
        prep_train_config = prep_train_config,
        run_train_config = run_train_config,
        prep_explore_config = prep_explore_config,
        run_explore_config = run_explore_config,
        prep_fp_config = prep_fp_config,
        run_fp_config = run_fp_config,
        select_confs_config = select_confs_config,
        collect_data_config = collect_data_config,
        cl_step_config = cl_step_config,
        upload_python_package = upload_python_package,
    )
    scheduler = make_naive_exploration_scheduler(config, old_style=old_style)

    type_map = config['type_map'] if old_style else config['inputs']['type_map']
    numb_models = config['numb_models'] if old_style else config['train']['numb_models']
    template_script = config['default_training_param'] if old_style else config['train']['template_script']
    train_config = {} if old_style else config['train']['config']
    lmp_config = config.get('lmp_config', {}) if old_style else config['explore']['config']
    fp_config = config.get('fp_config', {}) if old_style else config['fp']['config']
    kspacing, kgamma = get_kspacing_kgamma_from_incar(config['fp_incar'] if old_style else config['fp']['incar'])
    fp_pp_files = config['fp_pp_files'] if old_style else config['fp']['pp_files']
    incar_file = config['fp_incar'] if old_style else config['fp']['incar']
    fp_inputs = VaspInputs(
        kspacing = kspacing,
        kgamma = kgamma,
        incar_template_name = incar_file,
        potcar_names = fp_pp_files,
    )
    init_data_prefix = config.get('init_data_prefix') if old_style else config['inputs']['init_data_prefix']
    init_data = config['init_data_sys'] if old_style else config['inputs']['init_data_sys']
    if init_data_prefix is not None:
        init_data = [os.path.join(init_data_prefix, ii) for ii in init_data_sys]
    if isinstance(init_data,str):
        init_data = expand_sys_str(init_data)
    init_data = upload_artifact(init_data)
    iter_data = upload_artifact([])
    if init_models_paths is not None:
        init_models = upload_artifact(init_models_paths)
    else:
        init_models = None

    # here the scheduler is passed as input parameter to the concurrent_learning_op
    dpgen_step = Step(
        'dpgen-step',
        template = concurrent_learning_op,
        parameters = {
            "type_map" : type_map,
            "numb_models" : numb_models,
            "template_script" : template_script,
            "train_config" : train_config,
            "lmp_config" : lmp_config,
            "fp_config" : fp_config,
            'fp_inputs' : fp_inputs,
            "exploration_scheduler" : scheduler,
        },
        artifacts = {
            "init_models" : init_models,
            "init_data" : init_data,
            "iter_data" : iter_data,
        },
    )
    return dpgen_step


def wf_global_workflow(
        wf_config,
):
    workflow_config_from_dict(wf_config)

    # lebesgue context
    from dflow.plugins.lebesgue import LebesgueContext
    lb_context_config = wf_config.get("lebesgue_context_config", None)
    if lb_context_config:
        lebesgue_context = LebesgueContext(
            **lb_context_config,
        )
    else :
        lebesgue_context = None

    return lebesgue_context


def submit_concurrent_learning(
        wf_config,
        reuse_step = None,
        old_style = False,
):
    wf_config = normalize_submit_args(wf_config)

    context = wf_global_workflow(wf_config)
    
    dpgen_step = workflow_concurrent_learning(wf_config, old_style=old_style)

    wf = Workflow(name="dpgen", context=context)
    wf.add(dpgen_step)

    wf.submit(reuse_step=reuse_step)

    return wf


def print_list_steps(
        steps,
):
    ret = []
    for idx,ii in enumerate(steps):
        ret.append(f'{idx:8d}    {ii}')
    return '\n'.join(ret)


def expand_idx (in_list) :
    ret = []
    for ii in in_list :
        if isinstance(ii, int) :
            ret.append(ii)
        elif isinstance(ii, str):
            step_str = ii.split(':')
            if len(step_str) > 1 :
                step = int(step_str[1])
            else :
                step = 1
            range_str = step_str[0].split('-')
            if len(range_str) == 2:
                ret += range(int(range_str[0]), int(range_str[1]), step)
            elif len(range_str) == 1 :
                ret += [int(range_str[0])]
            else:
                raise RuntimeError('not expected range string', step_str[0])
    ret = sorted(list(set(ret)))
    return ret


def successful_step_keys(wf):
    all_step_keys_ = wf.query_keys_of_steps()
    wf_info = wf.query()
    all_step_keys = []
    for ii in all_step_keys_:
        if wf_info.get_step(key=ii)[0]['phase'] == 'Succeeded':
            all_step_keys.append(ii)
    return all_step_keys


def resubmit_concurrent_learning(
        wf_config,
        wfid,
        list_steps = False,
        reuse = None,
        old_style = False,
):
    wf_config = normalize_submit_args(wf_config)

    context = wf_global_workflow(wf_config)

    old_wf = Workflow(id=wfid)

    all_step_keys = successful_step_keys(old_wf)
    all_step_keys = sort_slice_ops(
        all_step_keys, ['run-train', 'run-lmp', 'run-fp'],)
    if list_steps:
        prt_str = print_keys_in_nice_format(
            all_step_keys, ['run-train', 'run-lmp', 'run-fp'],)
        print(prt_str)

    if reuse is None:
        return None
    reuse_idx = expand_idx(reuse)
    reuse_step = []
    old_wf_info = old_wf.query()
    for ii in reuse_idx:
        reuse_step += old_wf_info.get_step(key=all_step_keys[ii])

    wf = submit_concurrent_learning(
        wf_config, 
        reuse_step=reuse_step,
        old_style=old_style,
    )

    return wf
