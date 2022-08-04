#!/bin/python3

import types
import lammps
import numpy as np


class lmp:
    _lmp = None

    @staticmethod
    def cmd(command):
        if isinstance(command, str):
            lmp._lmp.command(command)
        elif isinstance(command, types.FunctionType):
            def wrapper(*args, **kwargs) -> None:
                lmp._lmp.command(command(*args, **kwargs))
            return wrapper

    @staticmethod
    def scmd(command):
        if isinstance(command, str):
            lmp._lmp.commands_string(command)
        elif isinstance(command, types.FunctionType):
            def wrapper(*args, **kwargs) -> None:
                lmp._lmp.commands_string(command(*args, **kwargs))
            return wrapper

    @staticmethod
    def lcmd(command):
        if isinstance(command, str):
            lmp._lmp.commands_list(command)
        elif isinstance(command, types.FunctionType):
            def wrapper(*args, **kwargs) -> None:
                lmp._lmp.commands_list(command(*args, **kwargs))
            return wrapper

    @staticmethod
    def start() -> None:
        lmp._lmp = lammps.lammps()
        lmp._lmp.command('package omp 12')

    @staticmethod
    def close() -> None:
        lmp._lmp.close()
        lmp._lmp = None

    @staticmethod
    def run(steps):
        lmp.cmd(f'run {steps}')
        
    @staticmethod
    def avcomp(comp_name):
        return lmp._lmp.numpy.extract_compute(
            comp_name, lammps.LMP_STYLE_ATOM, lammps.LMP_TYPE_VECTOR
        )

    @staticmethod
    def gvcomp(comp_name):
        return lmp._lmp.numpy.extract_compute(
            comp_name, lammps.LMP_STYLE_GLOBAL, lammps.LMP_TYPE_VECTOR
        )

    @staticmethod
    def gscomp(comp_name):
        return lmp._lmp.numpy.extract_compute(
            comp_name, lammps.LMP_STYLE_GLOBAL, lammps.LMP_TYPE_SCALAR
        )

    @staticmethod
    def aacomp(comp_name):
        return lmp._lmp.numpy.extract_compute(
            comp_name, lammps.LMP_STYLE_ATOM, lammps.LMP_TYPE_ARRAY
        )

    @staticmethod
    def evar(var_name):
        return lmp._lmp.numpy.extract_variable(
            var_name, vartype=lammps.LMP_VAR_EQUAL
        )

    @staticmethod
    def avar(var_name, group_name):
        return lmp._lmp.numpy.extract_variable(
            var_name, group=group_name, vartype=lammps.LMP_VAR_ATOM
        )


@lmp.scmd
def lmp_init():
    return """
units      metal
dimension  3
boundary   p p m
atom_style atomic
atom_modify map yes
"""


@lmp.scmd
def lmp_regions(si_lattice, width, si_top, si_bottom, si_fixed):
    return f"""
lattice diamond {si_lattice} orient x 1 0 0 orient y 0 1 0 orient z 0 0 1

region r_si_all     block {-width} {width} {-width} {width} {si_bottom} {si_top} units lattice
region r_fixed      block {-width} {width} {-width} {width} {si_bottom} {si_fixed} units lattice

region r_floor   block {-width}  {width}    {-width}  {width}    {si_fixed} {si_fixed+1}
region r_x_left  block {-width}  {-width+1} {-width}  {width}    {si_fixed} {si_top}
region r_x_right block {width-1} {width}    {-width}  {width}    {si_fixed} {si_top}
region r_y_left  block {-width}  {width}    {-width}  {-width+1} {si_fixed} {si_top}
region r_y_right block {-width}  {width}    {width-1} {width}    {si_fixed} {si_top}

region r_bath union 5 r_floor r_x_right r_x_left r_y_right r_y_left

region r_clusters block {-width} {width} {-width} {width} 0 INF units lattice
"""


@lmp.scmd
def lmp_add_fu(fu_x_coord, fu_y_coord, fu_z_coord):
    return f"""
molecule m_C60 ./mol.txt
create_atoms 1 single {fu_x_coord} {fu_y_coord} {fu_z_coord} mol m_C60 1 units box
"""


@lmp.scmd
def lmp_potentials():
    return """
pair_style  hybrid airebo/omp 3.0 tersoff/zbl/omp
pair_coeff  * * tersoff/zbl/omp SiC.tersoff.zbl Si C
pair_coeff  2 2 none
pair_coeff  * * airebo/omp CH.airebo NULL C
neighbor    3.0 bin
"""


@lmp.scmd
def lmp_groups():
    return """
group   g_fu     type 2
group   g_si_all type 1
group   g_fixed region r_fixed
group   g_nve subtract all g_fixed

group   g_thermostat dynamic g_si_all region r_bath
"""


@lmp.scmd
def lmp_computes(z_coord_threshold):
    return f"""
# compute ke per atom
compute atom_ke all ke/atom

# voronoi
compute   voro_occupation g_si_all voronoi/atom occupation only_group
variable  is_vacancy atom "c_voro_occupation[1]==0"
variable  vacancy_id atom "v_is_vacancy*id"
compute   vacancies g_si_all reduce sum v_is_vacancy

# sputtered atoms
variable is_sputtered atom "z>{z_coord_threshold}"
compute   sputter_all  all       reduce sum v_is_sputtered
compute   sputter_si   g_si_all  reduce sum v_is_sputtered
compute   sputter_c    g_fu      reduce sum v_is_sputtered
"""


@lmp.scmd
def lmp_thermo():
    return """
reset_timestep 0
timestep       0.001
thermo         10
thermo_style   custom step pe ke etotal temp c_vacancies dt time c_sputter_all c_sputter_c c_sputter_si
"""


@lmp.scmd
def lmp_fixes(temperature):
    return f"""
fix f_1 g_nve nve/omp
fix f_2 g_thermostat temp/berendsen {temperature} {temperature} 0.001 
fix f_3 all electron/stopping 10.0 ./elstop-table.txt region r_si_all
fix f_4 all dt/reset 1 0.0005 0.001 0.1
"""


@lmp.scmd
def lmp_clusters():
    return f"""
#group g_clusters_parent region r_clusters
#group g_clusters dynamic g_clusters_parent var is_sputtered
group g_clusters variable is_sputtered
compute clusters g_clusters cluster/atom 3
compute mass g_clusters property/atom mass

dump d_clusters g_clusters custom 20 {lmp.RESULTS_DIR}/clusters.dump id x y z vx vy vz type c_clusters c_atom_ke
dump d_all all custom 20 {lmp.RESULTS_DIR}/all.dump id x y z vx vy vz type c_clusters c_atom_ke
"""


@lmp.scmd
def lmp_sputtered_clusters(clusters):
    commands = ''
    for cluster_id in clusters:
        var = f'is_cluster_{cluster_id}'
        group = f'g_cluster_{cluster_id}'
        commands += f'variable {var} atom "c_clusters=={cluster_id}"\n'
        commands += f'group {group} variable {var}\n'
        commands += f'compute {cluster_id}_c g_fu reduce sum v_{var}\n'
        commands += f'compute {cluster_id}_si g_si_all reduce sum v_{var}\n'
        smom = f'{cluster_id}_mom'
        commands += f'compute {smom} {group} momentum\n'
        commands += f'compute {cluster_id}_mass {group} reduce sum c_mass\n'
        commands += f'variable {cluster_id}_ek equal "(c_{smom}[1]^2+c_{smom}[2]^2+c_{smom}[3]^2)/(2*c_{cluster_id}_mass)"\n'
        commands += f'variable {cluster_id}_angle equal "atan(c_{smom}[3]/sqrt(c_{smom}[1]^2+c_{smom}[2]^2))"\n'
    return commands


def get_clusters_table(clusters, cluster_ids):
    table = np.array([])
    for cluster_id in cluster_ids:
        comp_c = lmp.gscomp(f'{cluster_id}_c')
        comp_si = lmp.gscomp(f'{cluster_id}_si')
        comp_mom = lmp.gvcomp(f'{cluster_id}_mom')
        comp_mass = lmp.gscomp(f'{cluster_id}_mass')
        var_ek = lmp.evar(f'{cluster_id}_ek')
        var_angle = lmp.evar(f'{cluster_id}_angle')
        table = np.concatenate(
            (table, np.array([cluster_id, comp_si, comp_c, comp_mass, *comp_mom,
                              2*5.1875*1e-5*var_ek, 90-var_angle*180/np.pi]))
        )
    table = table.reshape((table.shape[0]//9,9))
    return table


def get_clusters_mask(atom_x, atom_cluster):
    mask_1 = (atom_cluster != 0)
    cluster_ids = set(np.unique(atom_cluster[mask_1]).flatten())

    mask_2 = (atom_x[:,2] < 2.0)
    no_cluster_ids = set(np.unique(atom_cluster[mask_2]).flatten())
    cluster_ids = list(cluster_ids.difference(no_cluster_ids))

    mask = np.isin(atom_cluster, cluster_ids)
    return mask, np.asarray(cluster_ids).astype(int)


def get_rim_info(group_ids, fu_x_coord, fu_y_coord):
    lmp.cmd("group g_rim id " + " ".join(group_ids.astype(int).astype(str)))
    lmp.cmd(f'variable r_rim atom "sqrt((x-{fu_x_coord})^2+(y-{fu_y_coord})^2)"')
    lmp.cmd('compute r_rim_sum g_rim reduce sum v_r_rim')
    lmp.cmd('compute r_rim_max g_rim reduce max v_r_rim')
    r_max = lmp.gscomp('r_rim_max')
    r_mean = lmp.gscomp('r_rim_sum') / len(group_ids)
    lmp.cmd('compute rim_z_sum g_rim reduce sum z')
    lmp.cmd('compute rim_z_max g_rim reduce max z')
    z_mean = lmp.gscomp('rim_z_sum') / len(group_ids)
    z_max = lmp.gscomp('rim_z_max')
    lmp.cmd('variable rim_count equal "count(g_rim)"')
    rim_count = lmp.evar('rim_count')

    return np.array([[rim_count, r_mean, r_max, z_mean, z_max]])


def get_crater_info(clusters):
    crater_id = np.bincount(clusters.astype(int)).argmax()
    lmp.cmd(f'variable is_crater atom "c_clusters=={crater_id}"')

    lmp.cmd('compute crater_num g_vac reduce sum v_is_crater')
    crater_count = lmp.gscomp('crater_num')
    voronoi = lmp.aacomp('voro_vol')
    cell_vol = np.median(voronoi, axis=0)[0]
    crater_vol = cell_vol * crater_count

    lmp.cmd('variable is_surface atom "z>-2.4*0.707"')
    lmp.cmd('compute surface_count g_vac reduce sum v_is_surface')
    surface_count = lmp.gscomp('surface_count')
    cell_surface = 7.3712
    surface_area = cell_surface * surface_count

    lmp.cmd('compute crater_z_mean g_vac reduce sum z')
    lmp.cmd('compute crater_z_min g_vac reduce min z')
    crater_z_min = lmp.gscomp('crater_z_min')
    crater_z_mean = lmp.gscomp('crater_z_mean') / crater_count
    
    return np.array([[crater_count,crater_vol,surface_area,crater_z_mean,crater_z_min]])


def get_carbon_hist():
    
    return None


def main():
    lmp.RESULTS_DIR = './results'
    
    lmp.start()

    fu_z_coord = 15
    fu_x_coord = 0
    fu_y_coord = 0

    fu_z_vel = 462.8

    si_lattice = 5.43
    si_bottom = -20
    si_fixed = si_bottom + 0.5
    si_top = 0

    temperature = 1e-6

    width = 20

    z_coord_threshold = 0.4

    lmp_init()
    lmp.cmd('read_data input.data')

    lmp_regions(si_lattice, width, si_top, si_bottom, si_fixed)
    lmp.cmd('write_restart restart.lammps')

    lmp_add_fu(fu_x_coord, fu_y_coord, fu_z_coord)
    lmp_potentials()
    lmp_groups()

    lmp_computes(z_coord_threshold)
    lmp_thermo()
    lmp_fixes(temperature)

    lmp.cmd(f'dump d_1 all custom 20 {lmp.RESULTS_DIR}/norm.dump id type xs ys zs')
    lmp.cmd(f'velocity g_fu set NULL NULL {-fu_z_vel} sum yes units box')
    lmp.run(100)

    lmp_clusters()
    lmp.cmd('run 1')

    vac_ids = lmp.avar('vacancy_id', 'g_si_all')
    vac_ids = vac_ids[vac_ids != 0]
    vac_group_command = "group g_vac id " + " ".join(vac_ids.astype(int).astype(str))

    atom_cluster = lmp.avcomp('clusters')
    atom_x = lmp._lmp.numpy.extract_atom('x')
    atom_id = lmp._lmp.numpy.extract_atom('id')
    atom_type = lmp._lmp.extract_atom('type')
    mask, cluster_ids = get_clusters_mask(atom_x, atom_cluster)

    lmp_sputtered_clusters(cluster_ids)
    clusters_table = get_clusters_table(atom_cluster[mask], cluster_ids)
    np.savetxt(f'{lmp.RESULTS_DIR}/clusters_table.txt', clusters_table, header='cluster_id N_Si N_C mass Px Py Pz Ek angle', delimiter='\t', fmt="%.5f")
    rim_info = get_rim_info(atom_id[~mask & (atom_cluster!=0)], fu_x_coord, fu_y_coord)
    np.savetxt(f'{lmp.RESULTS_DIR}/rim_table.txt', rim_info, header='N r_mean r_max z_mean z_max', delimiter='\t', fmt="%.5f")

    carbon_hist = get_carbon_hist(atom_x, atom_type)
    
    lmp.close()
    lmp.start()
    lmp.cmd('read_restart restart.lammps')
    lmp_potentials()
    lmp.cmd(vac_group_command)
    lmp.cmd('group g_si_all type 1')
    lmp.cmd('compute voro_vol g_si_all voronoi/atom only_group')
    lmp.cmd('compute clusters g_vac cluster/atom 3')
    lmp.cmd(f'dump d_clusters g_vac custom 20 {lmp.RESULTS_DIR}/crater.dump id x y z vx vy vz type c_clusters')
    lmp.cmd('run 1')

    clusters = lmp.avcomp('clusters')
    clusters = clusters[clusters != 0]
    crater_info = get_crater_info(clusters)
    np.savetxt(f'{lmp.RESULTS_DIR}/crater_table.txt', crater_info, header='N V S z_mean z_min', delimiter='\t', fmt="%.5f")
    
    lmp.close()


if __name__ == '__main__':
    main()
