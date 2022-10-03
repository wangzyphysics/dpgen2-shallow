import unittest, json, shutil, os, json, textwrap
from utils.context import dpgen2
import numpy as np
import random
import tempfile
import dpdata
from pathlib import Path
from dpgen2.entrypoint.submit_args import (
    normalize,
)
from dpgen2.utils import (
    normalize_step_dict,
)
from dpgen2.op import RunDPTrain

class TestArgs(unittest.TestCase):
    def test(self):
        old_data = json.loads(old_str)
        new_data = normalize(json.loads(new_str))
        default_config = normalize_step_dict(old_data.get('default_config', {}))
        self.assertEqual(old_data['model_devi_jobs'], new_data['explore']['stages'])
        self.assertEqual(old_data['sys_configs'], new_data['explore']['configurations'])
        self.assertEqual(old_data.get('sys_prefix'), new_data['explore']['configuration_prefix'])
        self.assertEqual(old_data['mass_map'], new_data['inputs']['mass_map'])
        self.assertEqual(old_data['type_map'], new_data['inputs']['type_map'])
        self.assertEqual(old_data['numb_models'], new_data['train']['numb_models'])
        self.assertEqual(old_data['fp_task_max'], new_data['fp']['task_max'])
        self.assertEqual(old_data['conv_accuracy'], new_data['explore']['conv_accuracy'])
        self.assertEqual(old_data['max_numb_iter'], new_data['explore']['max_numb_iter'])
        self.assertEqual(old_data.get('fatal_at_max', True), new_data['explore']['fatal_at_max'])
        self.assertEqual(old_data['model_devi_f_trust_lo'], new_data['explore']['f_trust_lo'])
        self.assertEqual(old_data['model_devi_f_trust_hi'], new_data['explore']['f_trust_hi'])
        self.assertEqual(old_data.get('model_devi_v_trust_lo'), new_data['explore']['v_trust_lo'])
        self.assertEqual(old_data.get('model_devi_v_trust_hi'), new_data['explore']['v_trust_hi'])
        self.assertEqual(old_data.get('train_style', 'dp'), new_data['train']['type'])
        self.assertEqual(old_data.get('explore_style', 'lmp'), new_data['explore']['type'])
        self.assertEqual(old_data.get('fp_style', 'vasp'), new_data['fp']['type'])
        self.assertEqual(normalize_step_dict(old_data.get('prep_train_config', default_config)), new_data['step_configs']['prep_train_config'])
        self.assertEqual(normalize_step_dict(old_data.get('run_train_config', default_config)), new_data['step_configs']['run_train_config'])
        self.assertEqual(normalize_step_dict(old_data.get('prep_explore_config', default_config)), new_data['step_configs']['prep_explore_config'])
        self.assertEqual(normalize_step_dict(old_data.get('run_explore_config', default_config)), new_data['step_configs']['run_explore_config'])
        self.assertEqual(normalize_step_dict(old_data.get('prep_fp_config', default_config)), new_data['step_configs']['prep_fp_config'])
        self.assertEqual(normalize_step_dict(old_data.get('run_fp_config', default_config)), new_data['step_configs']['run_fp_config'])
        self.assertEqual(normalize_step_dict(old_data.get('select_confs_config', default_config)), new_data['step_configs']['select_confs_config'])
        self.assertEqual(normalize_step_dict(old_data.get('collect_data_config', default_config)), new_data['step_configs']['collect_data_config'])
        self.assertEqual(normalize_step_dict(old_data.get('cl_step_config', default_config)), new_data['step_configs']['cl_step_config'])
        self.assertEqual(old_data.get('upload_python_package', None), new_data['upload_python_package'])
        self.assertEqual(old_data['type_map'], new_data['inputs']['type_map'])
        self.assertEqual(old_data['numb_models'], new_data['train']['numb_models'])
        self.assertEqual(old_data['default_training_param'], new_data['train']['template_script'])
        self.assertEqual(RunDPTrain.normalize_config({}), new_data['train']['config'])
        self.assertEqual(old_data.get('lmp_config', {}), new_data['explore']['config'])
        self.assertEqual(old_data.get('fp_config', {}), new_data['fp']['config'])
        self.assertEqual(old_data['fp_pp_files'], new_data['fp']['pp_files'])
        self.assertEqual(old_data['fp_incar'], new_data['fp']['incar'])
        self.assertEqual(old_data.get('init_data_prefix'), new_data['inputs']['init_data_prefix'])
        self.assertEqual(old_data['init_data_sys'], new_data['inputs']['init_data_sys'])


old_str = textwrap.dedent("""
{
    "train_style" : "dp",
    "explore_style" : "lmp",
    "fp_style" : "vasp",

    "default_config" : {
	"template_config" : {
	    "image" : "dflow:1.1.4",
	    "_comment" : "all"
	},
	"_comment" : "all"
    },

    "run_train_config" : {
	"template_config" : {
	    "image" : "deepmd-kit:wanghan",
	    "_comment" : "all"
	},
	"executor" : {
	    "type" : "lebesgue_v2",
	    "extra" : {
		"scass_type": "c6_m64_1 * NVIDIA 3090",
		"machine_type": "c6_m64_1 * NVIDIA 3090",
		"platform": "paratera",
		"program_id": "xxxx",
		"job_type": "container",
		"region" : "default"
	    }
	},
	"_comment" : "all"
    },
    "run_explore_config" : {
	"template_config" : {
	    "image" : "deepmd-kit:wanghan",
	    "_comment" : "all"
	},
	"executor" : {
	    "type" : "lebesgue_v2",
	    "extra" : {
		"scass_type": "c8_m32_cpu",
		"machine_type": "c8_m32_cpu",
		"platform": "paratera",
		"program_id": "xxxx",
		"job_type": "container",
		"region" : "default"
	    }
	},
	"_comment" : "all"
    },
    "run_fp_config" : {
	"template_config" : {
	    "image" : "vasp:wanghan",
	    "_comment" : "all"
	},
	"executor" : {
	    "type" : "lebesgue_v2",
	    "extra" : {
		"scass_type": "c16_m64_cpu",
		"machine_type": "c16_m64_cpu",
		"platform": "paratera",
		"program_id": "xxxx",
		"job_type": "container",
		"region" : "default"
	    }
	},
	"_comment" : "all"
    },

    "lmp_config": {
	"command": "lmp -var restart 0"
    },
    "fp_config": {
	"command": "source /opt/intel/oneapi/setvars.sh && mpirun -n 16 vasp_std",
        "log" : "vasp.log",
        "out" : "data"
    },

    "dflow_config" : {
	"host" : "http://60.205.112.9:2746",
	"s3_endpoint" : "60.205.112.9:9000",
	"_catalog_file_name" : "dflow"
    },

    "lebesgue_context_config": {
	"executor" : "lebesgue_v2",
        "extra" : {
	    "scass_type": "c8_m32_cpu",
	    "machine_type": "c8_m32_cpu",
            "platform": "xxxx",
            "program_id": "xxxx",
            "job_type": "container",
	    "region" : "default"
        },
        "app_name" : "Default",
        "org_id" : "xxx",
        "user_id" : "xxx",
        "tag" : ""
    },

    "_comment" : "upload the dpgen2 package if it is not in the images",
    "upload_python_package" : "/path/to/dpgen2",

    "max_numb_iter" :	5,
    "conv_accuracy" :	0.9,
    "fatal_at_max" :	false,

    "type_map":		["Al", "Mg"],
    "mass_map":		[27, 24],

    "init_data_prefix":	"",
    "init_data_sys":	[
	"init/al.fcc.01x01x01/02.md/sys-0004/deepmd",
	"init/mg.fcc.01x01x01/02.md/sys-0004/deepmd"
    ],
    "sys_configs_prefix": "", 
    "sys_configs":	[
	{
	    "lattice" : ["fcc", 4.57],
	    "replicate" : [2, 2, 2],
	    "numb_confs" : 30,
	    "concentration" : [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]
	}
    ],

    "_comment":		" 00.train ",
    "numb_models":	4,
    "default_training_param" : {
	"model" : {
	    "type_map":		["Al", "Mg"],
	    "descriptor": {
		"type":		"se_a",
		"sel":		[90, 90],
		"rcut_smth":	1.80,
		"rcut":		6.00,
		"neuron":	[25, 50, 100],
		"resnet_dt":	false,
		"axis_neuron":	4,
		"seed":		1
	    },
	    "fitting_net" : {
		"neuron":	[128, 128, 128],
		"resnet_dt":	true,
		"seed":		1
	    }
	},

	"loss" : {
	    "start_pref_e":	0.02,
	    "limit_pref_e":	1,
	    "start_pref_f":	1000,
	    "limit_pref_f":	1,
	    "start_pref_v":	0,
	    "limit_pref_v":	0
	},

	"learning_rate" : {
	    "start_lr":		0.001,
	    "stop_lr":		1e-8,
            "decay_steps":	100
	},

	"training" : {
	    "training_data": {
		"systems": [],
		"batch_size":"auto"
	    },
	    "numb_steps":1000,
	    "seed":10,
	    "disp_file":"lcurve.out",
	    "disp_freq":100,
	    "save_freq":1000
	}
    },

    "_comment":		" 01.model_devi ",
    "_comment": "model_devi_skip: the first x of the recorded frames",
    "model_devi_f_trust_lo":	0.05,
    "model_devi_f_trust_hi":	0.50,
    "model_devi_jobs":	[
	{ "_idx": 0, "ensemble": "nvt", "nsteps": 20, "press": [1.0,2.0], "sys_idx": [0], "temps": [50,100], "trj_freq": 10, "n_sample" : 3 }
    ],

    "_comment":		" 02.fp ",    
    "fp_style":		"vasp",
    "fp_task_max":	2,
    "fp_pp_files":	{"Al" : "vasp/POTCAR.Al", "Mg" : "vasp/POTCAR.Mg"},
    "fp_incar":         "vasp/INCAR",
    "_comment":		" that's all "
}
""")



new_str = textwrap.dedent("""
{
    "dflow_config" : {
	"host" : "http://address.of.the.host:port"
    },
    "dflow_s3_config" : {
	"s3_endpoint" : "address.of.the.s3.sever:port"
    },

    "lebesgue_context_config": {
	"executor" : "lebesgue_v2",
        "extra" : {
	    "scass_type": "c8_m32_cpu",
	    "machine_type": "c8_m32_cpu",
            "platform": "xxxx",
            "program_id": "xxxx",
            "job_type": "container",
	    "region" : "default"
        },
        "app_name" : "Default",
        "org_id" : "xxx",
        "user_id" : "xxx",
        "tag" : ""
    },

    "default_step_config" : {
	"template_config" : {
	    "image" : "dflow:1.1.4",
	    "_comment" : "all"
	},
	"_comment" : "all"
    },

    "step_configs":{
	"run_train_config" : {
	    "template_config" : {
		"image" : "deepmd-kit:wanghan",
		"_comment" : "all"
	    },
	    "executor" : {
		"type" : "lebesgue_v2",
		"extra" : {
		    "scass_type": "c6_m64_1 * NVIDIA 3090",
		    "machine_type": "c6_m64_1 * NVIDIA 3090",
		    "platform": "paratera",
		    "program_id": "xxxx",
		    "job_type": "container",
		    "region" : "default"
		}
	    },
	    "_comment" : "all"
	},
	"run_explore_config" : {
	    "template_config" : {
		"image" : "deepmd-kit:wanghan",
		"_comment" : "all"
	    },
	    "executor" : {
		"type" : "lebesgue_v2",
		"extra" : {
		    "scass_type": "c8_m32_cpu",
		    "machine_type": "c8_m32_cpu",
		    "platform": "paratera",
		    "program_id": "xxxx",
		    "job_type": "container",
		    "region" : "default"
		}
	    },
	    "_comment" : "all"
	},
	"run_fp_config" : {
	    "template_config" : {
		"image" : "vasp:wanghan",
		"_comment" : "all"
	    },
	    "executor" : {
		"type" : "lebesgue_v2",
		"extra" : {
		    "scass_type": "c16_m64_cpu",
		    "machine_type": "c16_m64_cpu",
		    "platform": "paratera",
		    "program_id": "xxxx",
		    "job_type": "container",
		    "region" : "default"
		}
	    },
	    "_comment" : "all"
	},
	"_comment" : "all"
    },

    "upload_python_package" : "/path/to/dpgen2",

    "inputs": {
	"type_map":		["Al", "Mg"],
	"mass_map":		[27, 24],
	"init_data_prefix":	"",
	"init_data_sys":	[
	    "init/al.fcc.01x01x01/02.md/sys-0004/deepmd",
	    "init/mg.fcc.01x01x01/02.md/sys-0004/deepmd"
	],
	"_comment" : "all"
    },
    "train":{
	"type" :	"dp",
	"numb_models" : 4,
	"config" : {},
	"template_script" : {
	    "model" : {
		"type_map":		["Al", "Mg"],
		"descriptor": {
		    "type":		"se_a",
		    "sel":		[90, 90],
		    "rcut_smth":	1.80,
		    "rcut":		6.00,
		    "neuron":	[25, 50, 100],
		    "resnet_dt":	false,
		    "axis_neuron":	4,
		    "seed":		1
		},
		"fitting_net" : {
		    "neuron":	[128, 128, 128],
		    "resnet_dt":	true,
		    "seed":		1
		}
	    },
	    "loss" : {
		"start_pref_e":	0.02,
		"limit_pref_e":	1,
		"start_pref_f":	1000,
		"limit_pref_f":	1,
		"start_pref_v":	0,
		"limit_pref_v":	0
	    },
	    "learning_rate" : {
		"start_lr":		0.001,
		"stop_lr":		1e-8,
		"decay_steps":	100
	    },
	    "training" : {
		"training_data": {
		    "systems": [],
		    "batch_size":"auto"
		},
		"numb_steps":1000,
		"seed":10,
		"disp_file":"lcurve.out",
		"disp_freq":100,
		"save_freq":1000
	    },
	    "_comment" : "all"
	},
	"_comment" : "all"
    },

    "explore" : {
	"type" : "lmp",
	"config" : {
	    "command": "lmp -var restart 0"
	},
	"max_numb_iter" :	5,
	"conv_accuracy" :	0.9,
	"fatal_at_max" :	false,
	"f_trust_lo":		0.05,
	"f_trust_hi":		0.50,
	"configuration_prefix": null, 
	"configuration":	[
	    {
		"lattice" : ["fcc", 4.57],
		"replicate" : [2, 2, 2],
		"numb_confs" : 30,
		"concentration" : [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]
	    }
	],
	"stages":	[
	    { "_idx": 0, "ensemble": "nvt", "nsteps": 20, "press": [1.0,2.0], "sys_idx": [0], "temps": [50,100], "trj_freq": 10, "n_sample" : 3 
	    }
	],
	"_comment" : "all"
    },
    "fp" : {
	"type" :	"vasp",
	"config" : {
	    "command": "source /opt/intel/oneapi/setvars.sh && mpirun -n 16 vasp_std"
	},
	"task_max":	2,
	"pp_files":	{"Al" : "vasp/POTCAR.Al", "Mg" : "vasp/POTCAR.Mg"},
	"incar":         "vasp/INCAR",
	"_comment" : "all"
    }
}
""")
