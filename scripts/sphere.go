package main

import (
	"bufio"
	"log"
	"math"
	"os"
	"strconv"
	"strings"
)

const (
	FLOAT_PREC = 5
)

type Dump struct {
	data      map[string]map[int][]float64
	timesteps []int
}

func NewDump(path string) Dump {
	file, err := os.Open(path)
	if err != nil {
		log.Fatalln("reading dump file:", path, err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	dump := Dump{
		data:      make(map[string]map[int][]float64),
		timesteps: make([]int, 0),
	}
	keys := make([]string, 0)

	is_read_atom_count := false
	is_read_atoms := false
	is_read_timestep := false

	atom_count := -1
	current_atom := -1
	current_timestep := -1
	for scanner.Scan() {
		if scanner.Text() == "ITEM: TIMESTEP" {
			is_read_atoms = false
			current_atom = -1
			is_read_timestep = true
			continue
		}

		if is_read_timestep {
			is_read_timestep = false
			current_timestep = parse_int(scanner.Text())
			dump.timesteps = append(dump.timesteps, current_timestep)
			// if current_timestep == 500 {
			// 	return dump
			// }
			log.Println("reading timestep:", current_timestep)
			continue
		}

		if scanner.Text() == "ITEM: NUMBER OF ATOMS" {
			is_read_atom_count = true
			continue
		}

		if is_read_atom_count {
			atom_count = parse_int(scanner.Text())
			is_read_atom_count = false
			continue
		}

		if strings.Contains(scanner.Text(), "ITEM: ATOMS") {
			is_read_atoms = true
			if len(keys) == 0 {
				for _, key := range strings.Split(scanner.Text(), " ")[2:] {
					keys = append(keys, key)
					dump.data[key] = make(map[int][]float64)
				}
			}

			for _, val := range dump.data {
				val[current_timestep] = make([]float64, atom_count)
			}
			continue
		}

		if is_read_atoms {
			tokens := strings.Split(scanner.Text(), " ")
			current_atom++
			for i, key := range keys {
				dump.data[key][current_timestep][current_atom] = parse_float(tokens[i])
			}
		}
	}

	if err := scanner.Err(); err != nil {
		log.Fatalln("reading dump file:", err)
	}

	return dump
}

func (d Dump) extract(field string, timestep int) []float64 {
	return d.data[field][timestep]
}

func parse_float(s string) float64 {
	f64, err := strconv.ParseFloat(s, 64)
	if err != nil {
		log.Fatalln("parsing float64: ", s, err)
	}
	return f64
}

func parse_int(s string) int {
	i, err := strconv.Atoi(s)
	if err != nil {
		log.Fatalln("parsing int: ", s, err)
	}
	return i
}

func calc_zero_lvl(dump Dump, timestep int) float64 {
	zero_lvl := math.SmallestNonzeroFloat64
	z := dump.extract("z", timestep)
	type_ := dump.extract("type", timestep)

	for i := range z {
		if z[i] > zero_lvl && type_[i] == 1 {
			zero_lvl = z[i]
		}
	}

	return zero_lvl
}

func calc_cluster_center(dump Dump, timestep int) (float64, float64) {
	var c_x, c_y, count float64
	x := dump.extract("x", timestep)
	y := dump.extract("y", timestep)
	type_ := dump.extract("type", timestep)

	for i := range x {
		if type_[i] == 2 {
			c_x += x[i]
			c_y += y[i]
			count++
		}
	}

	return c_x / count, c_y / count
}

func sphere_tables(dump Dump, zero_lvl, c_x, c_y float64) ([][]string, [][]string) {
	tab_ek := make([][]string, len(dump.timesteps)+1)
	tab_count := make([][]string, len(dump.timesteps)+1)
	R_start := 10
	R_stop := 30
	d_R := 5
	for i := range tab_ek {
		tab_ek[i] = make([]string, (R_stop-R_start)/d_R+2)
		tab_count[i] = make([]string, (R_stop-R_start)/d_R+2)
	}
	tab_ek[0][0] = "timestep\\R"
	tab_count[0][0] = "timestep\\R"

	for R := R_start; R <= R_stop; R += d_R {
		log.Println("table Ek - R:", R)
		j := (R-R_start)/d_R + 1
		tab_ek[0][j] = strconv.Itoa(R)
		tab_count[0][j] = strconv.Itoa(R)
		for i, timestep := range dump.timesteps {
			var sum_ek float64
			count := 0
			x := dump.extract("x", timestep)
			y := dump.extract("y", timestep)
			z := dump.extract("z", timestep)
			ek := dump.extract("c_atom_ke", timestep)
			for a_i := range x {
				dx := x[a_i] - c_y
				dy := y[a_i] - c_x
				dz := z[a_i] - zero_lvl
				m := dx*dx + dy*dy + dz*dz
				if m <= float64(R*R) && z[a_i] <= zero_lvl+1 {
					sum_ek += ek[a_i]
					count++
				}
			}
			tab_ek[i+1][j] = strconv.FormatFloat(float64(sum_ek), 'f', FLOAT_PREC, 64)
			tab_ek[i+1][0] = strconv.Itoa(timestep)
			tab_count[i+1][j] = strconv.Itoa(count)
			tab_count[i+1][0] = strconv.Itoa(timestep)
		}
	}

	return tab_ek, tab_count
}

func velocity_table(dump Dump, zero_lvl, c_x, c_y float64) [][]string {
	tab_vel := make([][]string, len(dump.timesteps)+1)
	vel := make([][]int, len(dump.timesteps))
	var cyllinder_r float64 = 20
	var cyllinder_h float64 = 10
	var bin_width float64 = 10
	var bin_start float64 = -180
	var bin_end float64 = 180
	bin_length := (bin_end - bin_start)
	bin_count := int(bin_length / bin_width)

	for i := range tab_vel {
		tab_vel[i] = make([]string, bin_count+1)
		if i != 0 {
			tab_vel[i][0] = strconv.Itoa(dump.timesteps[i-1])
			vel[i-1] = make([]int, len(tab_vel[i])-1)
		}
	}
	tab_vel[0][0] = "timestep\\deg"
	for i := range bin_count {
		tab_vel[0][i+1] = strconv.FormatFloat(float64(i)*bin_width+bin_start+bin_width/2, 'f', FLOAT_PREC, 64)
	}

	for j, timestep := range dump.timesteps {
		x := dump.extract("x", timestep)
		y := dump.extract("y", timestep)
		z := dump.extract("z", timestep)
		vx := dump.extract("vx", timestep)
		vy := dump.extract("vy", timestep)
		vz := dump.extract("vz", timestep)
		for i := range x {
			dx := x[i] - c_x
			dy := y[i] - c_y
			dz := z[i] - zero_lvl
			m := dx*dx + dy*dy
			if m <= cyllinder_r*cyllinder_r && math.Abs(dz) <= cyllinder_h {
				v_len := math.Sqrt(vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i])
				angle := math.Acos(vz[i]/v_len) * 180.0 / math.Pi
				if v_len < 10 {
					continue
				}
				if float64(dx*vx[i]+dy*vy[i]) > 0 {
					angle = (-1) * angle
				}
				if angle < bin_start {
					continue
				}
				if angle > bin_end {
					continue
				}
				index := int((angle - bin_start) / bin_width)
				if index == bin_count {
					index--
				}
				vel[j][index]++
			}
		}
	}

	for i, row := range vel {
		for j, cel := range row {
			tab_vel[i+1][j+1] = strconv.Itoa(cel)
		}
	}

	return tab_vel
}

func write_table(tab [][]string, path string) {
	f, err := os.Create(path)
	if err != nil {
		log.Fatalln("can't open file: ", path)
	}
	defer f.Close()
	for _, row := range tab {
		for j, cell := range row {
			if j > 0 {
				f.WriteString("\t")
			}
			f.WriteString(cell)
		}
		f.WriteString("\n")
	}
}

func main() {
	args := os.Args[1:]

	if len(args) != 1 {
		log.Fatal("usage: sphere.go RUN_DIR")
	}
	run_dir := args[0]
	dump_during := run_dir + "/dump.during"
	log.Println("using run_dir:", run_dir)

	dump := NewDump(dump_during)
	zero_lvl := calc_zero_lvl(dump, 0)
	center_x, center_y := calc_cluster_center(dump, 0)
	log.Printf("zero_lvl: %f, center: (%f, %f)\n", zero_lvl, center_x, center_y)

	sum_ek_path := run_dir + "/sphere_sum_ek.txt"
	count_path := run_dir + "/sphere_count.txt"
	tab_sum_ek, tab_count := sphere_tables(dump, zero_lvl, center_x, center_y)
	write_table(tab_sum_ek, sum_ek_path)
	write_table(tab_count, count_path)

	vel_path := run_dir + "/vel_distrib.txt"
	tab_vel := velocity_table(dump, zero_lvl, center_x, center_y)
	write_table(tab_vel, vel_path)
}
