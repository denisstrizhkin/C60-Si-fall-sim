#!/bin/python3

from os import path
import lammps
import numpy as np


class LAMMPS(lammps.lammps):
    def run(self, steps):
        self.command(f"run {steps}")

    def get_atom_vector_compute(self, comp_name):
        return self.numpy.extract_compute(
            comp_name, lammps.LMP_STYLE_ATOM, lammps.LMP_TYPE_VECTOR
        )

    def get_global_vector_comp(self, comp_name):
        return self.numpy.extract_compute(
            comp_name, lammps.LMP_STYLE_GLOBAL, lammps.LMP_TYPE_VECTOR
        )

    def get_global_scalar_comp(self, comp_name):
        return self.numpy.extract_compute(
            comp_name, lammps.LMP_STYLE_GLOBAL, lammps.LMP_TYPE_SCALAR
        )

    def get_atom_array_comp(self, comp_name):
        return self.numpy.extract_compute(
            comp_name, lammps.LMP_STYLE_ATOM, lammps.LMP_TYPE_ARRAY
        )

    def get_equal_variable(self, var_name):
        return self.numpy.extract_variable(var_name, vartype=lammps.LMP_VAR_EQUAL)

    def get_atom_variable(self, var_name, group_name):
        return self.numpy.extract_variable(
            var_name, group=group_name, vartype=lammps.LMP_VAR_ATOM
        )


class SIMULATION:
    def __init__(self, temperature, zero_lvl):
        self.lmp = None
        self.num_threads = 1

        self.temperature = temperature
        self.zero_lvl = zero_lvl

    def lmp_start(self):
        self.lmp = LAMMPS()
        self.lmp.command(f"package omp {self.num_threads}")

    def lmp_stop(self):
        self.lmp.close()
        self.lmp = None

    def set_input_file(self, input_file_path):
        self.input_file_paht = input_file_path

    def set_sim_num(self, sim_num):
        self.sim_num = sim_num

    def set_results_dir(self, results_dir_path):
        self.results_dir = results_dir_path

        def write_header(header_str, table_path):
            with open(table_path, "w", encoding="utf-8") as f:
                f.write("# " + header_str + "\n")

        def results_dir_join(file_path):
            return path.join(self.results_dir, file_path)

        self.clusters_table = results_dir_join("clusters_table.txt")
        self.rim_table = results_dir_join("rim_table.txt")
        self.carbon_table = results_dir_join("carbon_table.txt")
        self.crater_table = results_dir_join("crater_table.txt")
        self.carbon_dist = results_dir_join("carbon_dist.txt")

        write_header("sim_num N_Si N_C mass Px Py Pz Ek angle", self.clusters_table)
        write_header("sim_num N r_mean r_max z_mean z_max", self.rim_table)
        write_header("sim_num N r_mean r_max", self.carbon_table)
        write_header("sim_num N V S z_mean z_min", self.crater_table)
        write_header("z count", self.carbon_dist)

    def set_si_vars(self, si_bottom, si_top, si_width, si_lattice):
        self.si_lattice = si_lattice
        self.si_bottom = si_bottom
        self.si_top = si_top
        self.si_width = si_width
        self.si_fixed = si_bottom + 0.5

    def set_fu_vars(self, fu_x, fu_y, fu_z, fu_energy):
        self.fu_x_coord = fu_x
        self.fu_y_coord = fu_y
        self.fu_z_coord = self.si_top * self.si_lattice + fu_z
        self.fu_speed = fu_energy / 5.174

    def run(self):
        self.lmp_start()
        self.init()
        self.lmp.command(f"read_data {self.input_file_path}")

        self.regions()
        self.vacancies_restart_file = "./restart.lammps"
        self.lmp.cmd(f"write_restart {self.vacancies_restart_file}")

        simulation.add_fu()
        simulation.potentials()
        simulation.groups()

        simulation_computes()
        simulation_thermo()
        simulation_fixes(temperature)

        simulation.cmd(
            f"dump d_1 all custom 20 {lmp.RESULTS_DIR}/norm_{lmp.sim_num}.dump \
id t    ype xs ys zs"
        )
        lmp.cmd(f"velocity g_fu set NULL NULL {-fu_z_vel} sum yes units box")
        lmp.run(10000)

        lmp_recalc_zero_lvl(width, si_lattice)
        lmp_clusters()
        lmp.cmd("run 1")

        vac_ids = lmp.avar("vacancy_id", "g_si_all")
        vac_ids = vac_ids[vac_ids != 0]
        vac_group_command = "group g_vac id " + " ".join(
            vac_ids.astype(int).astype(str)
        )

        atom_cluster = lmp.avcomp("clusters")
        atom_x = self._lmp.numpy.extract_atom("x")
        atom_id = self._lmp.numpy.extract_atom("id")
        atom_type = self._lmp.numpy.extract_atom("type")
        mask, cluster_ids = get_clusters_mask(atom_x, atom_cluster)

        clusters_table = get_clusters_table(cluster_ids)
        append_table(lmp.CLUSTERS_TABLE, clusters_table)
        rim_info = get_rim_info(
            atom_id[~mask & (atom_cluster != 0)], fu_x_coord, fu_y_coord
        )
        append_table(lmp.RIM_TABLE, rim_info)

        carbon_hist = get_carbon_hist(atom_x, atom_type, mask)
        append_table(lmp.CARBON_DIST, carbon_hist, header=str(lmp.sim_num))
        carbon_info = get_carbon_info(
            atom_id[~mask & (atom_type == 2)], fu_x_coord, fu_y_coord
        )
        append_table(lmp.CARBON_TABLE, carbon_info)

        lmp.close()
        lmp.start(12)
        lmp.cmd("read_restart restart.lammps")
        lmp_potentials()
        lmp.cmd(vac_group_command)
        lmp.cmd("group g_si_all type 1")
        lmp.cmd("compute voro_vol g_si_all voronoi/atom only_group")
        lmp.cmd("compute clusters g_vac cluster/atom 3")
        lmp.cmd(
            f"dump d_clusters g_vac custom 20 {lmp.RESULTS_DIR}/crater_{lmp.sim_num}.dump \
id x     y z vx vy vz type c_clusters"
        )
        lmp.cmd("run 1")

        clusters = lmp.avcomp("clusters")
        clusters = clusters[clusters != 0]
        crater_info = get_crater_info(clusters)
        append_table(lmp.CRATER_TABLE, crater_info)

        lmp.close()

    def init(self):
        self.lmp.commands_string(
            f"""
units       metal
dimension   3
boundary    p p m
atom_style  atomic
atom_modify map yes
"""
        )

    def regions(self):
        W = self.width
        self.lmp.commands_string(
            f"""
lattice diamond {self.si_lattice} orient x 1 0 0 orient y 0 1 0 orient z 0 0 1

region si_all block {-self.width} {self.width} {-self.width} {self.width} {si_bottom} \
{si_top} units lattice

region fixed block {-self.width} {self.width} {-self.width} {self.width} {si_bottom} \
{si_fixed} units lattice

region floor   block {-self.width}  {self.width}    {-self.width}  {self.width}    {si_fixed} \
{si_fixed+1}
region x_left  block {-self.width}  {-self.width+1} {-self.width}  {self.width}    {si_fixed} \
{si_top}
region x_right block {self.width-1} {self.width}    {-self.width}  {self.width}    {si_fixed} \
{si_top}
region y_left  block {-self.width}  {self.width}    {-self.width}  {-self.width+1} {si_fixed} \
{si_top}
region y_right block {-self.width}  {self.width}    {self.width-1} {self.width}    {si_fixed} \
{si_top}

region bath union 5 floor x_right x_left y_right y_left

region clusters block {-self.width} {self.width} {-self.width} {self.width} 0 INF units lattice

region not_outside block {-self.width + 2} {self.width - 2} {-self.width + 2} \
    {self.width - 2} {si_bottom} {si_top+2} units lattice
"""
        )

    def lmp_add_fu(self):
        self._lmp.scmd(
            f"""
molecule m_C60 ./mol.txt
create_atoms 1 single {_fu_x_coord} {_fu_y_coord} {_fu_z_coord} \
mol m_C60 1 units box
"""
        )

    def lmp_potentials(self):
        self._lmp.scmd(
            """
pair_style  hybrid airebo/omp 3.0 tersoff/zbl/omp
pair_coeff  * * tersoff/zbl/omp SiC.tersoff.zbl Si C
pair_coeff  2 2 none
pair_coeff  * * airebo/omp CH.airebo NULL C
neighbor    3.0 bin
"""
        )

    def lmp_groups(self):
        self._lmp.scmd(
            """
group   fu     type 2
group   si_all type 1
group   fixed region fixed
group   nve subtract all fixed

group   thermostat dynamic si_all region bath

group   not_outside region not_outside
group   outside subtract si_all not_outside
"""
        )

    def lmp_computes(self):
        self._lmp.scmd(
            f"""
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
"""
        )

    def lmp_thermo(self):
        self._lmp.scmd(
            """
reset_timestep 0
timestep       0.001
thermo         10
thermo_style   custom step pe ke etotal temp c_vacancies dt time \
c_sputter_all c_sputter_c c_sputter_si
"""
        )

    def lmp_fixes():
        self._lmp.scmd(
            f"""
fix f_1 g_nve nve/omp
fix f_2 g_thermostat temp/berendsen {temperature} {temperature} 0.001
fix f_3 all electron/stopping 10.0 ./elstop-table.txt region r_si_all
fix f_4 all dt/reset 1 0.0005 0.001 0.1
"""
        )

    def lmp_clusters(self):
        self._lmp.scmd(
            f"""
variable is_sputtered delete
variable is_sputtered atom "z>{lmp.zero_lvl}"

group g_clusters variable is_sputtered
compute clusters g_clusters cluster/atom 3
compute mass g_clusters property/atom mass

dump d_clusters g_clusters custom 20 {lmp.RESULTS_DIR}/\
clusters_{lmp.sim_num}.dump id x y z vx vy vz type c_clusters c_atom_ke
dump d_all all custom 20 {lmp.RESULTS_DIR}/all_{lmp.sim_num}.dump \
id x y z vx vy vz type c_clusters c_atom_ke
"""
        )


def get_clusters_table(cluster_ids):
    table = np.array([])
    for cluster_id in cluster_ids:
        var = f"is_cluster_{cluster_id}"
        group = f"g_cluster_{cluster_id}"
        lmp.cmd(f'variable {var} atom "c_clusters=={cluster_id}"')
        lmp.cmd(f"group {group} variable {var}")
        lmp.cmd(f"compute {cluster_id}_c g_fu reduce sum v_{var}")
        lmp.cmd(f"compute {cluster_id}_si g_si_all reduce sum v_{var}")
        smom = f"{cluster_id}_mom"
        lmp.cmd(f"compute {smom} {group} momentum")
        lmp.cmd(f"compute {cluster_id}_mass {group} reduce sum c_mass")
        lmp.cmd(
            f'variable {cluster_id}_ek equal "(c_{smom}[1]^2+\
c_{smom}[2]^2+c_{smom}[3]^2)/(2*c_{cluster_id}_mass)"'
        )
        lmp.cmd(
            f'variable {cluster_id}_angle equal "atan(c_{smom}[3]/\
sqrt(c_{smom}[1]^2+c_{smom}[2]^2))"'
        )

        comp_c = lmp.gscomp(f"{cluster_id}_c")
        comp_si = lmp.gscomp(f"{cluster_id}_si")
        comp_mom = lmp.gvcomp(f"{cluster_id}_mom")
        comp_mass = lmp.gscomp(f"{cluster_id}_mass")
        var_ek = lmp.evar(f"{cluster_id}_ek")
        var_angle = lmp.evar(f"{cluster_id}_angle")
        table = np.concatenate(
            (
                table,
                np.array(
                    [
                        lmp.sim_num,
                        comp_si,
                        comp_c,
                        comp_mass,
                        *comp_mom,
                        2 * 5.1875 * 1e-5 * var_ek,
                        90 - var_angle * 180 / np.pi,
                    ]
                ),
            )
        )

        lmp.cmd(f"uncompute {cluster_id}_c")
        lmp.cmd(f"uncompute {cluster_id}_si")
        lmp.cmd(f"uncompute {cluster_id}_mass")
        lmp.cmd(f"uncompute {smom}")
        lmp.cmd(f"group {group} delete")

    table = table.reshape((table.shape[0] // 9, 9))
    return table


def get_clusters_mask(atom_x, atom_cluster):
    mask_1 = atom_cluster != 0
    cluster_ids = set(np.unique(atom_cluster[mask_1]).flatten())

    mask_2 = atom_x[:, 2] < (lmp.zero_lvl + 2.0)
    no_cluster_ids = set(np.unique(atom_cluster[mask_2]).flatten())
    cluster_ids = list(cluster_ids.difference(no_cluster_ids))

    mask = np.isin(atom_cluster, cluster_ids)
    return mask, np.asarray(cluster_ids).astype(int)


def get_rim_info(group_ids, fu_x_coord, fu_y_coord):
    lmp.cmd("group g_rim id " + " ".join(group_ids.astype(int).astype(str)))
    lmp.cmd(
        f'variable r_rim atom "sqrt((x-{fu_x_coord})^2+\
(y-{fu_y_coord})^2)"'
    )
    lmp.cmd("compute r_rim_sum g_rim reduce sum v_r_rim")
    lmp.cmd("compute r_rim_max g_rim reduce max v_r_rim")
    r_max = lmp.gscomp("r_rim_max")
    r_mean = lmp.gscomp("r_rim_sum") / len(group_ids)
    lmp.cmd("compute rim_z_sum g_rim reduce sum z")
    lmp.cmd("compute rim_z_max g_rim reduce max z")
    z_mean = lmp.gscomp("rim_z_sum") / len(group_ids)
    z_max = lmp.gscomp("rim_z_max")
    lmp.cmd('variable rim_count equal "count(g_rim)"')
    rim_count = lmp.evar("rim_count")

    return np.array(
        [
            [
                lmp.sim_num,
                rim_count,
                r_mean,
                r_max,
                z_mean - lmp.zero_lvl,
                z_max - lmp.zero_lvl,
            ]
        ]
    )


def get_crater_info(clusters):
    crater_id = np.bincount(clusters.astype(int)).argmax()
    lmp.cmd(f'variable is_crater atom "c_clusters=={crater_id}"')
    lmp.cmd("group g_vac clear")
    lmp.cmd("group g_vac variable is_crater")

    lmp.cmd("compute crater_num g_vac reduce sum v_is_crater")
    crater_count = lmp.gscomp("crater_num")
    voronoi = lmp.aacomp("voro_vol")
    cell_vol = np.median(voronoi, axis=0)[0]
    crater_vol = cell_vol * crater_count

    lmp.cmd(f'variable is_surface atom "z>-2.4*0.707+{lmp.zero_lvl}"')
    lmp.cmd("compute surface_count g_vac reduce sum v_is_surface")
    surface_count = lmp.gscomp("surface_count")
    cell_surface = 7.3712
    surface_area = cell_surface * surface_count

    lmp.cmd("compute crater_z_mean g_vac reduce sum z")
    lmp.cmd("compute crater_z_min g_vac reduce min z")
    crater_z_min = lmp.gscomp("crater_z_min") - lmp.zero_lvl
    crater_z_mean = lmp.gscomp("crater_z_mean") / crater_count - lmp.zero_lvl

    return np.array(
        [
            [
                lmp.sim_num,
                crater_count,
                crater_vol,
                surface_area,
                crater_z_mean,
                crater_z_min,
            ]
        ]
    )


def get_carbon_hist(atom_x, atom_type, mask):
    mask = (atom_type == 2) & ~mask
    z_coords = np.around(atom_x[mask][:, 2] - lmp.zero_lvl, 1)
    right = int(np.ceil(z_coords.max()))
    left = int(np.floor(z_coords.min()))
    hist, bins = np.histogram(z_coords, bins=(right - left), range=(left, right))
    length = len(hist)
    hist = np.concatenate(
        ((bins[1:] - 0.5).reshape(length, 1), hist.reshape(length, 1)), axis=1
    )

    return hist


def get_carbon_info(group_ids, fu_x_coord, fu_y_coord):
    lmp.cmd("group g_carbon id " + " ".join(group_ids.astype(int).astype(str)))
    lmp.cmd(
        f'variable r_carbon atom "sqrt((x-{fu_x_coord})^2+\
(y-{fu_y_coord})^2)"'
    )
    lmp.cmd("compute r_carbon_sum g_carbon reduce sum v_r_carbon")
    lmp.cmd("compute r_carbon_max g_carbon reduce max v_r_carbon")
    r_max = lmp.gscomp("r_carbon_max")
    r_mean = lmp.gscomp("r_carbon_sum") / len(group_ids)
    lmp.cmd('variable carbon_count equal "count(g_carbon)"')
    count = lmp.evar("carbon_count")

    return np.array([[lmp.sim_num, count, r_mean, r_max]])


def append_table(filename, table, header=""):
    with open(filename, "ab") as file:
        np.savetxt(file, table, delimiter="\t", fmt="%.5f", header=header)


def lmp_recalc_zero_lvl(width, lattice):
    lmp.cmd("compute max_outside_z outside reduce max z")
    max_outside_z = lmp.gscomp("max_outside_z")

    lmp.cmd(
        f"region surface block {-self.width} {self.width} {-self.width} {self.width} \
{(max_outside_z - 1.35)/lattice} {max_outside_z/lattice} units lattice"
    )
    lmp.cmd("group surface region surface")
    lmp.cmd("group outside_surface intersect surface outside")

    lmp.cmd("compute ave_outside_z outside_surface reduce ave z")
    ave_outside_z = lmp.gscomp("ave_outside_z")
    delta = max_outside_z - ave_outside_z
    lmp.zero_lvl = ave_outside_z + delta * 2


def get_fu_speed(energy):
    coeff = 5.174
    return coeff * np.sqrt(energy)


def main(simulation, fu_x_coord, fu_y_coord, fu_z_vel):
    simulation.start()
    simulation.init()
    simulation._lmp.cmd("read_data ./input_files/fall700.input.data")

    simulation.regions(si_lattice, self.width, si_top, si_bottom, si_fixed)
    simulation._lmp.cmd("write_restart restart.lammps")

    simulation.add_fu(fu_x_coord, fu_y_coord, fu_z_coord)
    simulation.potentials()
    simulation.groups()

    simulation_computes()
    simulation_thermo()
    simulation_fixes(temperature)

    simulation.cmd(
        f"dump d_1 all custom 20 {lmp.RESULTS_DIR}/norm_{lmp.sim_num}.dump \
id type xs ys zs"
    )
    lmp.cmd(f"velocity g_fu set NULL NULL {-fu_z_vel} sum yes units box")
    lmp.run(10000)

    lmp_recalc_zero_lvl(self.width, si_lattice)
    lmp_clusters()
    lmp.cmd("run 1")

    vac_ids = lmp.avar("vacancy_id", "g_si_all")
    vac_ids = vac_ids[vac_ids != 0]
    vac_group_command = "group g_vac id " + " ".join(vac_ids.astype(int).astype(str))

    atom_cluster = lmp.avcomp("clusters")
    atom_x = self._lmp.numpy.extract_atom("x")
    atom_id = self._lmp.numpy.extract_atom("id")
    atom_type = self._lmp.numpy.extract_atom("type")
    mask, cluster_ids = get_clusters_mask(atom_x, atom_cluster)

    clusters_table = get_clusters_table(cluster_ids)
    append_table(lmp.CLUSTERS_TABLE, clusters_table)
    rim_info = get_rim_info(
        atom_id[~mask & (atom_cluster != 0)], fu_x_coord, fu_y_coord
    )
    append_table(lmp.RIM_TABLE, rim_info)

    carbon_hist = get_carbon_hist(atom_x, atom_type, mask)
    append_table(lmp.CARBON_DIST, carbon_hist, header=str(lmp.sim_num))
    carbon_info = get_carbon_info(
        atom_id[~mask & (atom_type == 2)], fu_x_coord, fu_y_coord
    )
    append_table(lmp.CARBON_TABLE, carbon_info)

    lmp.close()
    lmp.start(12)
    lmp.cmd("read_restart restart.lammps")
    lmp_potentials()
    lmp.cmd(vac_group_command)
    lmp.cmd("group g_si_all type 1")
    lmp.cmd("compute voro_vol g_si_all voronoi/atom only_group")
    lmp.cmd("compute clusters g_vac cluster/atom 3")
    lmp.cmd(
        f"dump d_clusters g_vac custom 20 {lmp.RESULTS_DIR}/crater_{lmp.sim_num}.dump \
id x y z vx vy vz type c_clusters"
    )
    lmp.cmd("run 1")

    clusters = lmp.avcomp("clusters")
    clusters = clusters[clusters != 0]
    crater_info = get_crater_info(clusters)
    append_table(lmp.CRATER_TABLE, crater_info)

    lmp.close()


if __name__ == "__main__":
    # 0K - 83.19 | 700K - 83.391
    simulation = SIMULATION(temperature=700, zero_lvl=83.391)
    simulation.set_si_vars(si_bottom=-16, si_top=15.3, si_width=12, si_lattice=5.43)

    simulation.set_input_file("./input_files/fall700.input.data")
    simulation.set_results_dir("./results")

    def rand_coord():
        return simulation.si_lattice * (np.random.rand() * 2 - 1)

    for i in range(1):
        simulation.set_sim_num(i + 1)

        x = rand_coord()
        y = rand_coord()

        try:
            simulation.set_fu_vars(fu_energy=2000, fu_x=x, fu_y=y, fu_z=15)
            simulation.run()
        except lammps.MPIAbortException:
            pass
        except Exception as e:
            print(e)
            pass

    print("*** FINISHED COMPLETELY ***")
