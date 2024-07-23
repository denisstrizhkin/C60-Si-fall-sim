package main

import (
	"bufio"
	"log"
	"math"
	"os"
	"strconv"
	"strings"
)

type Dump struct {
	data      map[string]map[int][]float32
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
		data:      make(map[string]map[int][]float32),
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
					dump.data[key] = make(map[int][]float32)
				}
			}

			for _, val := range dump.data {
				val[current_timestep] = make([]float32, atom_count)
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

func (d Dump) extract(field string, timestep int) []float32 {
	return d.data[field][timestep]
}

func parse_float(s string) float32 {
	f64, err := strconv.ParseFloat(s, 32)
	if err != nil {
		log.Fatalln("parsing float32: ", s, err)
	}
	return float32(f64)
}

func parse_int(s string) int {
	i, err := strconv.Atoi(s)
	if err != nil {
		log.Fatalln("parsing int: ", s, err)
	}
	return i
}

func calc_zero_lvl(dump Dump, timestep int) float32 {
	var zero_lvl float32 = math.SmallestNonzeroFloat32
	z := dump.extract("z", timestep)
	type_ := dump.extract("type", timestep)

	for i := range z {
		if z[i] > zero_lvl && type_[i] == 1 {
			zero_lvl = z[i]
		}
	}

	return zero_lvl
}

func calc_cluster_center(dump Dump, timestep int) (float32, float32) {
	var c_x, c_y, count float32
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

func sphere_tables(dump Dump, zero_lvl, c_x, c_y float32) ([][]string, [][]string) {
	tab_ek := make([][]string, len(dump.timesteps)+1)
	tab_count := make([][]string, len(dump.timesteps)+1)
	R_start := 10
	R_stop := 30
	d_R := 5
	for i, _ := range tab_ek {
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
			var sum_ek float32
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
				if m <= float32(R*R) && z[a_i] <= zero_lvl+1 {
					sum_ek += ek[a_i]
					count++
				}
			}
			tab_ek[i+1][j] = strconv.FormatFloat(float64(sum_ek), 'f', -1, 32)
			tab_ek[i+1][0] = strconv.Itoa(timestep)
			tab_count[i+1][j] = strconv.Itoa(count)
			tab_count[i+1][0] = strconv.Itoa(timestep)
		}
	}

	return tab_ek, tab_count
}

func velocity_table(dump Dump, zero_lvl, c_x, c_y float32) [][]string {
	tab_vel := make([][]string, len(dump.timesteps)+1)
	vel := make([][]int, len(dump.timesteps))
	bin_start := -180
	bin_end := 180
	bin_width := 10
	R := 20
	height := 10
	for i := range tab_vel {
		tab_vel[i] = make([]string, (bin_end-bin_start)/bin_width+1)
		if i != 0 {
			tab_vel[i][0] = strconv.Itoa(dump.timesteps[i-1])
			vel[i-1] = make([]int, len(tab_vel[i])-1)
		}
	}
	tab_vel[0][0] = "timestep\\deg"
	for i := 5; i <= 175; i += 10 {
		tab_vel[0][i/10+1] = strconv.Itoa(i)
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
			if m <= float32(R*R) && math.Abs(float64(dz)) <= float64(height) {
				v_len := math.Sqrt(float64(vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i]))
				angle := math.Acos(float64(vz[i])/v_len) * 180.0 / math.Pi
				if float32(dx*vx + dy*vy) < 0 
					angle = (-1)*angle
				index := int(math.Round(angle))
				if index == bin_end {
					index--
				}
				index /= 10
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
