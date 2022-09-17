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

    def get_global_vector_compute(self, comp_name):
        return self.numpy.extract_compute(
            comp_name, lammps.LMP_STYLE_GLOBAL, lammps.LMP_TYPE_VECTOR
        )

    def get_global_scalar_compute(self, comp_name):
        return self.numpy.extract_compute(
            comp_name, lammps.LMP_STYLE_GLOBAL, lammps.LMP_TYPE_SCALAR
        )

    def get_atom_array_compute(self, comp_name):
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
    def __init__(self, temperature, zero_lvl, run_time, num_threads=1):
        self.lmp = None
        self.num_threads = num_threads

        self.temperature = temperature
        self.zero_lvl = zero_lvl
        self.run_time = run_time

    def lmp_start(self):
        self.lmp = LAMMPS()
        self.lmp.command(f"package omp {self.num_threads}")

    def lmp_stop(self):
        self.lmp.close()
        self.lmp = None

    def set_input_file(self, input_file_path):
        self.input_file_path = input_file_path

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
        self.si_fixed = self.si_bottom + 0.5

    def set_fu_vars(self, fu_x, fu_y, fu_z, fu_energy):
        self.fu_x_coord = fu_x
        self.fu_y_coord = fu_y
        self.fu_z_coord = self.si_top * self.si_lattice + fu_z
        self.fu_speed = np.sqrt(fu_energy) * 5.174

    def run(self):
        self.lmp_start()
        self.init()
        self.lmp.command(f"read_data {self.input_file_path}")

        self.regions()
        self.vacancies_restart_file = "./restart.lammps"
        self.lmp.command(f"write_restart {self.vacancies_restart_file}")

        self.add_fu()
        self.potentials()
        self.groups()

        self.computes()
        self.thermo()
        self.fixes()

        self.lmp.command(
            f"dump 1 all custom 2000 {self.results_dir}/norm_{self.sim_num}.dump \
id type xs ys zs"
        )
        self.lmp.command(
            f"velocity fu set NULL NULL {-self.fu_speed} sum yes units box"
        )
        self.lmp.run(self.run_time)

        self.recalc_zero_lvl()
        self.clusters()
        self.lmp.run(1)

        vac_ids = self.lmp.get_atom_variable("vacancy_id", "si_all")
        vac_ids = vac_ids[vac_ids != 0]
        vac_group_command = "group vac id " + " ".join(vac_ids.astype(int).astype(str))

        atom_cluster = self.lmp.get_atom_vector_compute("clusters")
        atom_x = self.lmp.numpy.extract_atom("x")
        atom_id = self.lmp.numpy.extract_atom("id")
        atom_type = self.lmp.numpy.extract_atom("type")
        mask, cluster_ids = self.get_clusters_mask(atom_x, atom_cluster)

        clusters_table = self.get_clusters_table(cluster_ids)
        self.append_table(self.clusters_table, clusters_table)
        rim_info = self.get_rim_info(atom_id[~mask & (atom_cluster != 0)])
        self.append_table(self.rim_table, rim_info)

        carbon_hist = self.get_carbon_hist(atom_x, atom_type, mask)
        self.append_table(self.carbon_dist, carbon_hist, header=str(self.sim_num))
        carbon_info = self.get_carbon_info(atom_id[~mask & (atom_type == 2)])
        self.append_table(self.carbon_table, carbon_info)

        self.lmp_stop()
        self.lmp_start()
        self.lmp.command(f"read_restart {self.vacancies_restart_file}")
        self.potentials()
        self.lmp.command(vac_group_command)
        self.lmp.command("group si_all type 1")
        self.lmp.command("compute voro_vol si_all voronoi/atom only_group")
        self.lmp.command("compute clusters vac cluster/atom 3")
        self.lmp.command(
            f"dump clusters vac custom 20 {self.results_dir}/crater_{self.sim_num}.dump \
id x     y z vx vy vz type c_clusters"
        )
        self.lmp.command("run 1")

        clusters = self.lmp.get_atom_vector_compute("clusters")
        clusters = clusters[clusters != 0]
        crater_info = self.get_crater_info(clusters)
        self.append_table(self.crater_table, crater_info)

        self.lmp.close()

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
        self.lmp.commands_string(
            f"""
lattice diamond {self.si_lattice} orient x 1 0 0 orient y 0 1 0 orient z 0 0 1

region si_all block {-self.si_width} {self.si_width} {-self.si_width} {self.si_width} {self.si_bottom} \
{self.si_top} units lattice

region fixed block {-self.si_width} {self.si_width} {-self.si_width} {self.si_width} {self.si_bottom} \
{self.si_fixed} units lattice

region floor   block {-self.si_width}  {self.si_width}    {-self.si_width}  {self.si_width}    {self.si_fixed} \
{self.si_fixed+1}
region x_left  block {-self.si_width}  {-self.si_width+1} {-self.si_width}  {self.si_width}    {self.si_fixed} \
{self.si_top}
region x_right block {self.si_width-1} {self.si_width}    {-self.si_width}  {self.si_width}    {self.si_fixed} \
{self.si_top}
region y_left  block {-self.si_width}  {self.si_width}    {-self.si_width}  {-self.si_width+1} {self.si_fixed} \
{self.si_top}
region y_right block {-self.si_width}  {self.si_width}    {self.si_width-1} {self.si_width}    {self.si_fixed} \
{self.si_top}

region bath union 5 floor x_right x_left y_right y_left

region clusters block {-self.si_width} {self.si_width} {-self.si_width} {self.si_width} 0 INF units lattice

region not_outside block {-self.si_width + 2} {self.si_width - 2} {-self.si_width + 2} \
    {self.si_width - 2} {self.si_bottom} {self.si_top+2} units lattice
"""
        )

    def add_fu(self):
        self.lmp.commands_string(
            f"""
molecule m_C60 ./mol.txt
create_atoms 1 single {self.fu_x_coord} {self.fu_y_coord} {self.fu_z_coord} \
mol m_C60 1 units box
"""
        )

    def potentials(self):
        self.lmp.commands_string(
            """
pair_style  hybrid airebo/omp 3.0 tersoff/zbl/omp
pair_coeff  * * tersoff/zbl/omp SiC.tersoff.zbl Si C
pair_coeff  2 2 none
pair_coeff  * * airebo/omp CH.airebo NULL C
neighbor    3.0 bin
"""
        )

    def groups(self):
        self.lmp.commands_string(
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

    def computes(self):
        self.lmp.commands_string(
            f"""
# compute ke per atom
compute atom_ke all ke/atom

# voronoi
compute   voro_occupation si_all voronoi/atom occupation only_group
variable  is_vacancy atom "c_voro_occupation[1]==0"
variable  vacancy_id atom "v_is_vacancy*id"
compute   vacancies si_all reduce sum v_is_vacancy

# sputtered atoms
variable is_sputtered atom "z>{self.zero_lvl}"
compute  sputter_all  all    reduce sum v_is_sputtered
compute  sputter_si   si_all reduce sum v_is_sputtered
compute  sputter_c    fu     reduce sum v_is_sputtered
"""
        )

    def thermo(self):
        self.lmp.commands_string(
            """
reset_timestep 0
timestep       0.001
thermo         10
thermo_style   custom step pe ke etotal temp c_vacancies dt time \
c_sputter_all c_sputter_c c_sputter_si
"""
        )

    def fixes(self):
        self.lmp.commands_string(
            f"""
fix f_1 nve nve/omp
fix f_2 thermostat temp/berendsen {self.temperature} {self.temperature} 0.001
fix f_3 all electron/stopping 10.0 ./elstop-table.txt region si_all
fix f_4 all dt/reset 1 0.0005 0.001 0.1
"""
        )

    def clusters(self):
        self.lmp.commands_string(
            f"""
variable is_sputtered delete
variable is_sputtered atom "z>{self.zero_lvl}"

group clusters variable is_sputtered
compute clusters clusters cluster/atom 3
compute mass clusters property/atom mass

dump clusters clusters custom 20 {self.results_dir}/\
clusters_{self.sim_num}.dump id x y z vx vy vz type c_clusters c_atom_ke
dump all all custom 20 {self.results_dir}/all_{self.sim_num}.dump \
id x y z vx vy vz type c_clusters c_atom_ke
"""
        )

    def get_clusters_table(self, cluster_ids):
        table = np.array([])
        for cluster_id in cluster_ids:
            var = f"is_cluster_{cluster_id}"
            group = f"cluster_{cluster_id}"
            self.lmp.command(f'variable {var} atom "c_clusters=={cluster_id}"')
            self.lmp.command(f"group {group} variable {var}")
            self.lmp.command(f"compute {cluster_id}_c fu reduce sum v_{var}")
            self.lmp.command(f"compute {cluster_id}_si si_all reduce sum v_{var}")
            smom = f"{cluster_id}_mom"
            self.lmp.command(f"compute {smom} {group} momentum")
            self.lmp.command(f"compute {cluster_id}_mass {group} reduce sum c_mass")
            self.lmp.command(
                f'variable {cluster_id}_ek equal "(c_{smom}[1]^2+\
               c_{smom}[2]^2+c_{smom}[3]^2)/(2*c_{cluster_id}_mass)"'
            )
            self.lmp.command(
                f'variable {cluster_id}_angle equal "atan(c_{smom}[3]/\
               sqrt(c_{smom}[1]^2+c_{smom}[2]^2))"'
            )

            comp_c = self.lmp.get_global_scalar_compute(f"{cluster_id}_c")
            comp_si = self.lmp.get_global_scalar_compute(f"{cluster_id}_si")
            comp_mom = self.lmp.get_global_vector_compute(f"{cluster_id}_mom")
            comp_mass = self.lmp.get_global_scalar_compute(f"{cluster_id}_mass")
            var_ek = self.lmp.get_equal_variable(f"{cluster_id}_ek")
            var_angle = self.lmp.get_equal_variable(f"{cluster_id}_angle")
            table = np.concatenate(
                (
                    table,
                    np.array(
                        [
                            self.sim_num,
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

            self.lmp.command(f"uncompute {cluster_id}_c")
            self.lmp.command(f"uncompute {cluster_id}_si")
            self.lmp.command(f"uncompute {cluster_id}_mass")
            self.lmp.command(f"uncompute {smom}")
            self.lmp.command(f"group {group} delete")

        table = table.reshape((table.shape[0] // 9, 9))
        return table

    def get_clusters_mask(self, atom_x, atom_cluster):
        mask_1 = atom_cluster != 0
        cluster_ids = set(np.unique(atom_cluster[mask_1]).flatten())

        mask_2 = atom_x[:, 2] < (self.zero_lvl + 2.0)
        no_cluster_ids = set(np.unique(atom_cluster[mask_2]).flatten())
        cluster_ids = list(cluster_ids.difference(no_cluster_ids))

        mask = np.isin(atom_cluster, cluster_ids)
        return mask, np.asarray(cluster_ids).astype(int)

    def get_rim_info(self, group_ids):
        self.lmp.command(
            "group g_rim id " + " ".join(group_ids.astype(int).astype(str))
        )
        self.lmp.command(
            f'variable r_rim atom "sqrt((x-{self.fu_x_coord})^2+\
           (y-{self.fu_y_coord})^2)"'
        )
        self.lmp.command("compute r_rim_sum g_rim reduce sum v_r_rim")
        self.lmp.command("compute r_rim_max g_rim reduce max v_r_rim")
        r_max = self.lmp.get_global_scalar_compute("r_rim_max")
        r_mean = self.lmp.get_global_scalar_compute("r_rim_sum") / len(group_ids)
        self.lmp.command("compute rim_z_sum g_rim reduce sum z")
        self.lmp.command("compute rim_z_max g_rim reduce max z")
        z_mean = self.lmp.get_global_scalar_compute("rim_z_sum") / len(group_ids)
        z_max = self.lmp.get_global_scalar_compute("rim_z_max")
        self.lmp.command('variable rim_count equal "count(g_rim)"')
        rim_count = self.lmp.get_equal_variable("rim_count")

        return np.array(
            [
                [
                    self.sim_num,
                    rim_count,
                    r_mean,
                    r_max,
                    z_mean - self.zero_lvl,
                    z_max - self.zero_lvl,
                ]
            ]
        )

    def get_crater_info(self, clusters):
        crater_id = np.bincount(clusters.astype(int)).argmax()
        self.lmp.command(f'variable is_crater atom "c_clusters=={crater_id}"')
        self.lmp.command("group vac clear")
        self.lmp.command("group vac variable is_crater")

        self.lmp.command("compute crater_num vac reduce sum v_is_crater")
        crater_count = self.lmp.get_global_scalar_compute("crater_num")
        voronoi = self.lmp.get_atom_array_compute("voro_vol")
        cell_vol = np.median(voronoi, axis=0)[0]
        crater_vol = cell_vol * crater_count

        self.lmp.command(f'variable is_surface atom "z>-2.4*0.707+{self.zero_lvl}"')
        self.lmp.command("compute surface_count vac reduce sum v_is_surface")
        surface_count = self.lmp.get_global_scalar_compute("surface_count")
        cell_surface = 7.3712
        surface_area = cell_surface * surface_count

        self.lmp.command("compute crater_z_mean vac reduce sum z")
        self.lmp.command("compute crater_z_min vac reduce min z")
        crater_z_min = (
            self.lmp.get_global_scalar_compute("crater_z_min") - self.zero_lvl
        )
        crater_z_mean = (
            self.lmp.get_global_scalar_compute("crater_z_mean") / crater_count
            - self.zero_lvl
        )

        return np.array(
            [
                [
                    self.sim_num,
                    crater_count,
                    crater_vol,
                    surface_area,
                    crater_z_mean,
                    crater_z_min,
                ]
            ]
        )

    def get_carbon_hist(self, atom_x, atom_type, mask):
        mask = (atom_type == 2) & ~mask
        z_coords = np.around(atom_x[mask][:, 2] - self.zero_lvl, 1)
        right = int(np.ceil(z_coords.max()))
        left = int(np.floor(z_coords.min()))
        hist, bins = np.histogram(z_coords, bins=(right - left), range=(left, right))
        length = len(hist)
        hist = np.concatenate(
            ((bins[1:] - 0.5).reshape(length, 1), hist.reshape(length, 1)), axis=1
        )

        return hist

    def get_carbon_info(self, group_ids):
        self.lmp.command(
            "group g_carbon id " + " ".join(group_ids.astype(int).astype(str))
        )
        self.lmp.command(
            f'variable r_carbon atom "sqrt((x-{self.fu_x_coord})^2+\
           (y-{self.fu_y_coord})^2)"'
        )
        self.lmp.command("compute r_carbon_sum g_carbon reduce sum v_r_carbon")
        self.lmp.command("compute r_carbon_max g_carbon reduce max v_r_carbon")
        r_max = self.lmp.get_global_scalar_compute("r_carbon_max")
        r_mean = self.lmp.get_global_scalar_compute("r_carbon_sum") / len(group_ids)
        self.lmp.command('variable carbon_count equal "count(g_carbon)"')
        count = self.lmp.get_equal_variable("carbon_count")

        return np.array([[self.sim_num, count, r_mean, r_max]])

    def append_table(self, filename, table, header=""):
        with open(filename, "ab") as file:
            np.savetxt(file, table, delimiter="\t", fmt="%.5f", header=header)

    def recalc_zero_lvl(self):
        self.lmp.command("variable outside_z atom z")
        outside_z = self.lmp.get_atom_variable("outside_z", "outside")
        outside_z = np.sort(outside_z)[-20:]
        max_outside_z = outside_z.mean()
        # self.lmp.command("compute max_outside_z outside reduce max z")
        # max_outside_z = self.lmp.get_global_scalar_compute("max_outside_z")

        self.lmp.command(
            f"region surface block {-self.si_width} {self.si_width} {-self.si_width} {self.si_width} \
           {(max_outside_z - 1.35)/self.si_lattice} {max_outside_z/self.si_lattice} units lattice"
        )
        self.lmp.command("group surface region surface")
        self.lmp.command("group outside_surface intersect surface outside")

        self.lmp.command("compute ave_outside_z outside_surface reduce ave z")
        ave_outside_z = self.lmp.get_global_scalar_compute("ave_outside_z")
        delta = max_outside_z - ave_outside_z
        self.zero_lvl = ave_outside_z + delta * 2

        print("max_outside_z:", max_outside_z)
        print("ave_outside_z:", ave_outside_z)
        print("delta:", delta)
        print("new zer_lvl:", self.zero_lvl)


def main():
    energy = 8_000
    run_time = int(energy * (5 / 4))

    # 0K    -  83.19
    # 300K  -  82.4535
    # 700K  -  83.391
    temperature = 0
    input_file_root = "./input_files"

    if temperature == 0:
        zero_lvl = 83.19
        input_file_path = path.join(input_file_root, "fall.input.data")
    elif temperature == 300:
        zero_lvl = 82.4535
        input_file_path = path.join(input_file_root, "fall300.input.data")
    elif temperature == 700:
        zero_lvl = 83.391
        input_file_path = path.join(input_file_root, "fall700.input.data")

    simulation = SIMULATION(
        temperature=temperature, zero_lvl=zero_lvl, run_time=run_time, num_threads=12
    )
    simulation.set_si_vars(si_bottom=-16, si_top=15.3, si_width=12, si_lattice=5.43)

    simulation.set_input_file(input_file_path)
    simulation.set_results_dir("./results")

    def rand_coord():
        return simulation.si_lattice * (np.random.rand() * 2 - 1)

    for i in range(1):
        simulation.set_sim_num(i + 1)

        x = rand_coord()
        y = rand_coord()

        try:
            simulation.set_fu_vars(fu_energy=energy, fu_x=x, fu_y=y, fu_z=15)
            simulation.run()
        except lammps.MPIAbortException:
            pass
        except Exception as e:
            print(e)
            pass

    print("*** FINISHED COMPLETELY ***")


if __name__ == "__main__":
    main()
