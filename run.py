#!/bin/python3

from os import path
import lammps
import numpy as np


class lmp:
    _lmp = None

    @staticmethod
    def cmd(command):
        lmp._lmp.command(command)

    @staticmethod
    def scmd(command):
        lmp._lmp.commands_string(command)

    @staticmethod
    def lcmd(command):
        lmp._lmp.commands_list(command)

    @staticmethod
    def start(num_of_threads) -> None:
        lmp._lmp = lammps.lammps()
        lmp._lmp.command(f'package omp {num_of_threads}')

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


def lmp_init():
    lmp.scmd("""
units      metal
dimension  3
boundary   p p m
atom_style atomic
atom_modify map yes
""")


def lmp_regions(si_lattice, width, si_top, si_bottom, si_fixed):
    lmp.scmd(f"""
lattice diamond {si_lattice} orient x 1 0 0 orient y 0 1 0 orient z 0 0 1

region r_si_all     block {-width} {width} {-width} {width} {si_bottom} \
{si_top} units lattice
region r_fixed      block {-width} {width} {-width} {width} {si_bottom} \
{si_fixed} units lattice

region r_floor   block {-width}  {width}    {-width}  {width}    {si_fixed} \
{si_fixed+1}
region r_x_left  block {-width}  {-width+1} {-width}  {width}    {si_fixed} \
{si_top}
region r_x_right block {width-1} {width}    {-width}  {width}    {si_fixed} \
{si_top}
region r_y_left  block {-width}  {width}    {-width}  {-width+1} {si_fixed} \
{si_top}
region r_y_right block {-width}  {width}    {width-1} {width}    {si_fixed} \
{si_top}

region r_bath union 5 r_floor r_x_right r_x_left r_y_right r_y_left

region r_clusters block {-width} {width} {-width} {width} 0 INF units lattice

region not_outside block {-width + 2} {width - 2} {-width + 2} \
    {width - 2} {si_bottom} {si_top+2} units lattice
""")


def lmp_add_fu(fu_x_coord, fu_y_coord, fu_z_coord):
    lmp.scmd(f"""
molecule m_C60 ./mol.txt
create_atoms 1 single {fu_x_coord} {fu_y_coord} {fu_z_coord} \
mol m_C60 1 units box
""")


def lmp_potentials():
    lmp.scmd("""
pair_style  hybrid airebo/omp 3.0 tersoff/zbl/omp
pair_coeff  * * tersoff/zbl/omp SiC.tersoff.zbl Si C
pair_coeff  2 2 none
pair_coeff  * * airebo/omp CH.airebo NULL C
neighbor    3.0 bin
""")


def lmp_groups():
    lmp.scmd("""
group   g_fu     type 2
group   g_si_all type 1
group   g_fixed region r_fixed
group   g_nve subtract all g_fixed

group   g_thermostat dynamic g_si_all region r_bath

group   not_outside region not_outside
group   outside subtract g_si_all not_outside
""")


def lmp_computes():
    lmp.scmd(f"""
# compute ke per atom
compute atom_ke all ke/atom

# voronoi
compute   voro_occupation g_si_all voronoi/atom occupation only_group
variable  is_vacancy atom "c_voro_occupation[1]==0"
variable  vacancy_id atom "v_is_vacancy*id"
compute   vacancies g_si_all reduce sum v_is_vacancy

# sputtered atoms
variable is_sputtered atom "z>{lmp.zero_lvl}"
compute   sputter_all  all       reduce sum v_is_sputtered
compute   sputter_si   g_si_all  reduce sum v_is_sputtered
compute   sputter_c    g_fu      reduce sum v_is_sputtered
""")


def lmp_thermo():
    lmp.scmd("""
reset_timestep 0
timestep       0.001
thermo         10
thermo_style   custom step pe ke etotal temp c_vacancies dt time \
c_sputter_all c_sputter_c c_sputter_si
""")


def lmp_fixes(temperature):
    lmp.scmd(f"""
fix f_1 g_nve nve/omp
fix f_2 g_thermostat temp/berendsen {temperature} {temperature} 0.001
fix f_3 all electron/stopping 10.0 ./elstop-table.txt region r_si_all
fix f_4 all dt/reset 1 0.0005 0.001 0.1
""")


def lmp_clusters():
    lmp.scmd(f"""
variable is_sputtered delete
variable is_sputtered atom "z>{lmp.zero_lvl}"

group g_clusters variable is_sputtered
compute clusters g_clusters cluster/atom 3
compute mass g_clusters property/atom mass

dump d_clusters g_clusters custom 20 {lmp.RESULTS_DIR}/\
clusters_{lmp.sim_num}.dump id x y z vx vy vz type c_clusters c_atom_ke
dump d_all all custom 20 {lmp.RESULTS_DIR}/all_{lmp.sim_num}.dump \
id x y z vx vy vz type c_clusters c_atom_ke
""")


def get_clusters_table(cluster_ids):
    table = np.array([])
    for cluster_id in cluster_ids:
        var = f'is_cluster_{cluster_id}'
        group = f'g_cluster_{cluster_id}'
        lmp.cmd(f'variable {var} atom "c_clusters=={cluster_id}"')
        lmp.cmd(f'group {group} variable {var}')
        lmp.cmd(f'compute {cluster_id}_c g_fu reduce sum v_{var}')
        lmp.cmd(f'compute {cluster_id}_si g_si_all reduce sum v_{var}')
        smom = f'{cluster_id}_mom'
        lmp.cmd(f'compute {smom} {group} momentum')
        lmp.cmd(f'compute {cluster_id}_mass {group} reduce sum c_mass')
        lmp.cmd(f'variable {cluster_id}_ek equal "(c_{smom}[1]^2+\
c_{smom}[2]^2+c_{smom}[3]^2)/(2*c_{cluster_id}_mass)"')
        lmp.cmd(f'variable {cluster_id}_angle equal "atan(c_{smom}[3]/\
sqrt(c_{smom}[1]^2+c_{smom}[2]^2))"')

        comp_c = lmp.gscomp(f'{cluster_id}_c')
        comp_si = lmp.gscomp(f'{cluster_id}_si')
        comp_mom = lmp.gvcomp(f'{cluster_id}_mom')
        comp_mass = lmp.gscomp(f'{cluster_id}_mass')
        var_ek = lmp.evar(f'{cluster_id}_ek')
        var_angle = lmp.evar(f'{cluster_id}_angle')
        table = np.concatenate(
            (table, np.array([lmp.sim_num, comp_si, comp_c, comp_mass,
                              *comp_mom,
                              2*5.1875*1e-5*var_ek, 90-var_angle*180/np.pi]))
        )

        lmp.cmd(f'uncompute {cluster_id}_c')
        lmp.cmd(f'uncompute {cluster_id}_si')
        lmp.cmd(f'uncompute {cluster_id}_mass')
        lmp.cmd(f'uncompute {smom}')
        lmp.cmd(f'group {group} delete')

    table = table.reshape((table.shape[0]//9, 9))
    return table


def get_clusters_mask(atom_x, atom_cluster):
    mask_1 = (atom_cluster != 0)
    cluster_ids = set(np.unique(atom_cluster[mask_1]).flatten())

    mask_2 = (atom_x[:, 2] < (lmp.zero_lvl + 2.0))
    no_cluster_ids = set(np.unique(atom_cluster[mask_2]).flatten())
    cluster_ids = list(cluster_ids.difference(no_cluster_ids))

    mask = np.isin(atom_cluster, cluster_ids)
    return mask, np.asarray(cluster_ids).astype(int)


def get_rim_info(group_ids, fu_x_coord, fu_y_coord):
    lmp.cmd("group g_rim id " + " ".join(group_ids.astype(int).astype(str)))
    lmp.cmd(f'variable r_rim atom "sqrt((x-{fu_x_coord})^2+\
(y-{fu_y_coord})^2)"')
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

    return np.array([[lmp.sim_num, rim_count, r_mean,
                      r_max, z_mean - lmp.zero_lvl, z_max - lmp.zero_lvl]])


def get_crater_info(clusters):
    crater_id = np.bincount(clusters.astype(int)).argmax()
    lmp.cmd(f'variable is_crater atom "c_clusters=={crater_id}"')
    lmp.cmd('group g_vac clear')
    lmp.cmd('group g_vac variable is_crater')

    lmp.cmd('compute crater_num g_vac reduce sum v_is_crater')
    crater_count = lmp.gscomp('crater_num')
    voronoi = lmp.aacomp('voro_vol')
    cell_vol = np.median(voronoi, axis=0)[0]
    crater_vol = cell_vol * crater_count

    lmp.cmd(f'variable is_surface atom "z>-2.4*0.707+{lmp.zero_lvl}"')
    lmp.cmd('compute surface_count g_vac reduce sum v_is_surface')
    surface_count = lmp.gscomp('surface_count')
    cell_surface = 7.3712
    surface_area = cell_surface * surface_count

    lmp.cmd('compute crater_z_mean g_vac reduce sum z')
    lmp.cmd('compute crater_z_min g_vac reduce min z')
    crater_z_min = lmp.gscomp('crater_z_min') - lmp.zero_lvl
    crater_z_mean = lmp.gscomp('crater_z_mean') / crater_count - lmp.zero_lvl

    return np.array([[lmp.sim_num, crater_count, crater_vol, surface_area,
                      crater_z_mean, crater_z_min]])


def get_carbon_hist(atom_x, atom_type, mask):
    mask = (atom_type == 2) & ~mask
    z_coords = np.around(atom_x[mask][:, 2]-lmp.zero_lvl, 1)
    right = int(np.ceil(z_coords.max()))
    left = int(np.floor(z_coords.min()))
    hist, bins = np.histogram(z_coords,
                              bins=(right-left), range=(left, right))
    length = len(hist)
    hist = np.concatenate(((bins[1:]-0.5).reshape(length, 1),
                           hist.reshape(length, 1)), axis=1)

    return hist


def get_carbon_info(group_ids, fu_x_coord, fu_y_coord):
    lmp.cmd("group g_carbon id " + " ".join(group_ids.astype(int).astype(str)))
    lmp.cmd(f'variable r_carbon atom "sqrt((x-{fu_x_coord})^2+\
(y-{fu_y_coord})^2)"')
    lmp.cmd('compute r_carbon_sum g_carbon reduce sum v_r_carbon')
    lmp.cmd('compute r_carbon_max g_carbon reduce max v_r_carbon')
    r_max = lmp.gscomp('r_carbon_max')
    r_mean = lmp.gscomp('r_carbon_sum') / len(group_ids)
    lmp.cmd('variable carbon_count equal "count(g_carbon)"')
    count = lmp.evar('carbon_count')

    return np.array([[lmp.sim_num, count, r_mean, r_max]])


def append_table(filename, table, header=''):
    with open(filename, 'ab') as file:
        np.savetxt(file, table, delimiter='\t', fmt='%.5f', header=header)


def lmp_recalc_zero_lvl(width,lattice):
    lmp.cmd('compute max_outside_z outside reduce max z')
    max_outside_z = lmp.gscomp('max_outside_z')

    lmp.cmd(f'region surface block {-width} {width} {-width} {width} \
{(max_outside_z - 1.35)/lattice} {max_outside_z/lattice} units lattice')
    lmp.cmd('group surface region surface')
    lmp.cmd('group outside_surface intersect surface outside')

    lmp.cmd('compute ave_outside_z outside_surface reduce ave z')
    ave_outside_z = lmp.gscomp('ave_outside_z')
    delta = max_outside_z - ave_outside_z
    lmp.zero_lvl = ave_outside_z + delta * 2;


def main(fu_x_coord, fu_y_coord, fu_z_vel):
    lmp.start(12)

    fu_z_coord = 15

    si_lattice = 5.43
    si_bottom = -16
    si_fixed = si_bottom + 0.5
    si_top = 15.3

    temperature = 1e-6

    width = 12

    # lmp.zero_lvl = 83.19 # 0K
    lmp.zero_lvl = 83.391  # 700K

    fu_z_coord += si_top * 5.43

    lmp_init()
    lmp.cmd('read_data ./input_files/fall700.input.data')

    lmp_regions(si_lattice, width, si_top, si_bottom, si_fixed)
    lmp.cmd('write_restart restart.lammps')

    lmp_add_fu(fu_x_coord, fu_y_coord, fu_z_coord)
    lmp_potentials()
    lmp_groups()

    lmp_computes()
    lmp_thermo()
    lmp_fixes(temperature)

    lmp.cmd(f'dump d_1 all custom 20 {lmp.RESULTS_DIR}/norm_{lmp.sim_num}.dump \
id type xs ys zs')
    lmp.cmd(f'velocity g_fu set NULL NULL {-fu_z_vel} sum yes units box')
    lmp.run(10000)

    lmp_recalc_zero_lvl(width,si_lattice)
    lmp_clusters()
    lmp.cmd('run 1')

    vac_ids = lmp.avar('vacancy_id', 'g_si_all')
    vac_ids = vac_ids[vac_ids != 0]
    vac_group_command = "group g_vac id " + " ".join(vac_ids.astype(int)
                                                     .astype(str))

    atom_cluster = lmp.avcomp('clusters')
    atom_x = lmp._lmp.numpy.extract_atom('x')
    atom_id = lmp._lmp.numpy.extract_atom('id')
    atom_type = lmp._lmp.numpy.extract_atom('type')
    mask, cluster_ids = get_clusters_mask(atom_x, atom_cluster)

    clusters_table = get_clusters_table(cluster_ids)
    append_table(lmp.CLUSTERS_TABLE, clusters_table)
    rim_info = get_rim_info(atom_id[~mask & (atom_cluster != 0)],
                            fu_x_coord, fu_y_coord)
    append_table(lmp.RIM_TABLE, rim_info)

    carbon_hist = get_carbon_hist(atom_x, atom_type, mask)
    append_table(lmp.CARBON_DIST, carbon_hist, header=str(lmp.sim_num))
    carbon_info = get_carbon_info(atom_id[~mask & (atom_type == 2)],
                                  fu_x_coord, fu_y_coord)
    append_table(lmp.CARBON_TABLE, carbon_info)

    lmp.close()
    lmp.start(12)
    lmp.cmd('read_restart restart.lammps')
    lmp_potentials()
    lmp.cmd(vac_group_command)
    lmp.cmd('group g_si_all type 1')
    lmp.cmd('compute voro_vol g_si_all voronoi/atom only_group')
    lmp.cmd('compute clusters g_vac cluster/atom 3')
    lmp.cmd(f'dump d_clusters g_vac custom 20 {lmp.RESULTS_DIR}/crater_{lmp.sim_num}.dump \
id x y z vx vy vz type c_clusters')
    lmp.cmd('run 1')

    clusters = lmp.avcomp('clusters')
    clusters = clusters[clusters != 0]
    crater_info = get_crater_info(clusters)
    append_table(lmp.CRATER_TABLE, crater_info)

    lmp.close()


if __name__ == '__main__':
    lmp.RESULTS_DIR = './results'
    lmp.CLUSTERS_TABLE = path.join(lmp.RESULTS_DIR, 'clusters_table.txt')
    lmp.RIM_TABLE = path.join(lmp.RESULTS_DIR, 'rim_table.txt')
    lmp.CARBON_TABLE = path.join(lmp.RESULTS_DIR, 'carbon_table.txt')
    lmp.CRATER_TABLE = path.join(lmp.RESULTS_DIR, 'crater_table.txt')
    lmp.CARBON_DIST = path.join(lmp.RESULTS_DIR, 'carbon_dist.txt')

    def write_header(header_str, table):
        with open(table, 'w', encoding='utf-8') as file:
            file.write('# ' + header_str + '\n')

    write_header('sim_num N_Si N_C mass Px Py Pz Ek angle', lmp.CLUSTERS_TABLE)
    write_header('sim_num N r_mean r_max z_mean z_max', lmp.RIM_TABLE)
    write_header('sim_num N r_mean r_max', lmp.CARBON_TABLE)
    write_header('sim_num N V S z_mean z_min', lmp.CRATER_TABLE)
    write_header('z count', lmp.CARBON_DIST)

    def rand_coord():
        return 5.43 * (np.random.rand() * 2 - 1)

    for i in range(1):
        lmp.sim_num = i+1

        x = rand_coord()
        y = rand_coord()

        try:
            main(x, y, 633.72)
        except lammps.MPIAbortException:
            pass
        except Exception as e:
            print(e)
            pass

    print('*** FINISHED COMPLETELY ***')
