from pathlib import (
    Path,
)
from typing import (
    List,
)

import numpy as np
from ase import (
    Atoms,
)
from ase.geometry import (
    cellpar_to_cell,
)
from ase.io import (
    read,
    write,
)
from deepmd.calculator import (
    DP,
)
from deepmd.infer import (
    calc_model_devi,
    DeepPot
)
from dflow.python import (
    OP,
    OPIO,
    Artifact,
    BigParameter,
    OPIOSign,
    Parameter,
)

from dpgen2.utils import (
    set_directory,
)


class RunCalyModelDevi(OP):
    r"""calculate model deviaion of trajectories structures.

    Structure optimization will be executed in `optim_path`. The trajectory
    will be stored in files `op["traj"]` and `op["model_devi"]`, respectively.

    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "type_map": BigParameter(List[str]),
                "task_name": Parameter(str),
                "traj_dirs": Artifact(List[Path]),
                "models": Artifact(List[Path]),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "task_name": Parameter(str),
                "traj": Artifact(Path),
                "model_devi": Artifact(Path),
            }
        )

    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        r"""Execute the OP.

        Parameters
        ----------
        ip : dict
            Input dict with components:
            - `type_map`: (`List[str]`) The type map of elements.
            - `task_name`: (`str`) The name of the task.
            - `traj_dirs`: (`Artifact(List[Path])`) The List of paths that contains trajectory files.
            - `models`: (`Artifact(List[Path])`) The frozen model to estimate the model deviation.

        Returns
        -------
        Any
            Output dict with components:
            - `task_name`: (`str`) The name of task.
            - `traj`: (`Artifact(Path)`) The output trajectory.
            - `model_devi`: (`Artifact(Path)`) The model deviation. The order of recorded model deviations should be consistent with the order of frames in `traj`.

        """
        type_map = ip["type_map"]

        models = ip["models"]
        print(models)
        all_models = [model.resolve() for model in models]
        print(all_models)
        # graphs = [DP(model).dp for model in all_models]
        graphs = [DeepPot(model).dp for model in all_models]

        work_dir = Path(ip["task_name"])

        traj_dirs = ip["traj_dirs"]
        traj_dirs = [traj_dir.resolve() for traj_dir in traj_dirs]

        dump_file_name = "traj.dump"
        model_devi_file_name = "model_devi.out"

        Devis = []
        tcount = 0
        with set_directory(work_dir):
            dump_file = Path().joinpath(dump_file_name)
            model_devi_file = Path().joinpath(model_devi_file_name)
            f = open(dump_file, "a")
            for traj_dir in traj_dirs:
                for traj_name in traj_dir.rglob("*.traj"):
                    atoms_list = parse_traj(traj_name)
                    if atoms_list is None:
                        continue
                    for atoms in atoms_list:
                        dump_str = atoms2lmpdump(atoms, tcount, type_map)
                        f.write(dump_str)
                        pbc = np.all(atoms.get_pbc())
                        coord = atoms.get_positions().reshape(1, -1)
                        cell = atoms.get_cell().reshape(1, -1) if pbc else None
                        atom_types = atoms.numbers
                        print(coord, cell, atom_types, graphs)
                        devi = calc_model_devi(coord, cell, atom_types, graphs)[0]
                        devi[0] = tcount
                        Devis.append(devi)
                        tcount += 1
            f.close()
            Devis = np.vstack(Devis)
            write_model_devi_out(Devis, model_devi_file)

        ret_dict = {
            "task_name": str(work_dir),
            "traj": work_dir / dump_file,
            "model_devi": work_dir / model_devi_file,
        }

        return OPIO(ret_dict)


def atoms2lmpdump(atoms, struc_idx, type_map):
    """down triangle cell can be obtained from
    cell params: a, b, c, alpha, beta, gamma.
    cell = cellpar_to_cell([a, b, c, alpha, beta, gamma])
    lx, ly, lz = cell[0][0], cell[1][1], cell[2][2]
    xy, xz, yz = cell[1][0], cell[2][0], cell[2][1]
    (lx,ly,lz) = (xhi-xlo,yhi-ylo,zhi-zlo)
    xlo_bound = xlo + MIN(0.0,xy,xz,xy+xz)
    xhi_bound = xhi + MAX(0.0,xy,xz,xy+xz)
    ylo_bound = ylo + MIN(0.0,yz)
    yhi_bound = yhi + MAX(0.0,yz)
    zlo_bound = zlo
    zhi_bound = zhi

    ref: https://docs.lammps.org/Howto_triclinic.html
    """
    dump_str = "ITEM: TIMESTEP\n"
    dump_str += f"{struc_idx}\n"
    dump_str += "ITEM: NUMBER OF ATOMS\n"
    dump_str += f"{atoms.get_global_number_of_atoms()}\n"

    cellpars = atoms.cell.cellpar()
    new_cell = cellpar_to_cell(cellpars)
    new_atoms = Atoms(
        numbers=atoms.numbers,
        cell=new_cell,
        scaled_positions=atoms.get_scaled_positions(),
    )

    xy, xz, yz = new_cell[1][0], new_cell[2][0], new_cell[2][1]
    xlo, ylo, zlo = 0, 0, 0
    lx, ly, lz = new_cell[0][0], new_cell[1][1], new_cell[2][2]
    xhi, yhi, zhi = lx + xlo, ly + ylo, lz + zlo
    xlo_bound = xlo + min(0.0, xy, xz, xy + xz)
    xhi_bound = xhi + max(0.0, xy, xz, xy + xz)
    ylo_bound = ylo + min(0.0, yz)
    yhi_bound = yhi + max(0.0, yz)
    zlo_bound = zlo
    zhi_bound = zhi

    dump_str += "ITEM: BOX BOUNDS xy xz yz pp pp pp\n"
    dump_str += "%20.10f %20.10f %20.10f\n" % (xlo_bound, xhi_bound, xy)
    dump_str += "%20.10f %20.10f %20.10f\n" % (ylo_bound, yhi_bound, xz)
    dump_str += "%20.10f %20.10f %20.10f\n" % (zlo_bound, zhi_bound, yz)
    dump_str += "ITEM: ATOMS id type x y z fx fy fz\n"
    for idx, atom in enumerate(new_atoms):
        type_id = type_map.index(atom.symbol) + 1
        dump_str += "%5d %5d" % (idx + 1, type_id)
        dump_str += "%20.10f %20.10f %20.10f" % (
            atom.position[0],
            atom.position[1],
            atom.position[2],
        )
        dump_str += "%20.10f %20.10f %20.10f\n" % (0, 0, 0)
    dump_str = dump_str.strip("\n")
    return dump_str


def parse_traj(traj_file) -> None | List[Atoms]:
    # optimization will at least return one structures in traj file
    trajs = read(traj_file, index=":", format="traj")

    numb_traj = len(trajs)
    assert numb_traj >= 1, "traj file is broken."

    origin = trajs[0]
    dis_mtx = origin.get_all_distances(mic=True)
    row, col = np.diag_indices_from(dis_mtx)
    dis_mtx[row, col] = np.nan
    is_reasonable = np.nanmin(dis_mtx) > 0.6

    if is_reasonable:
        if len(trajs) >= 20:
            selected_traj = [trajs[iii] for iii in [4, 9, -10, -5, -1]]
        elif 5 <= len(trajs) < 20:
            selected_traj = [
                trajs[np.random.randint(4, len(trajs) - 1)] for _ in range(4)
            ]
            selected_traj.append(trajs[-1])
        elif 3 <= len(trajs) < 5:
            selected_traj = [trajs[round((len(trajs) - 1) / 2)]]
            selected_traj.append(trajs[-1])
        elif len(trajs) == 2:
            selected_traj = [trajs[0], trajs[-1]]
        else:  # len(trajs) == 1
            selected_traj = [trajs[0]]
    else:
        selected_traj = None
    return selected_traj


def write_model_devi_out(devi: np.ndarray, fname: str, header: str = ""):
    assert devi.shape[1] == 7
    header = "%s\n%10s" % (header, "step")
    for item in "vf":
        header += "%19s%19s%19s" % (
            f"max_devi_{item}",
            f"min_devi_{item}",
            f"avg_devi_{item}",
        )
    with open(fname, "ab") as fp:
        np.savetxt(
            fp,
            devi,
            fmt=["%12d"] + ["%19.6e" for _ in range(devi.shape[1] - 1)],
            delimiter="",
            header=header,
        )
    return devi
