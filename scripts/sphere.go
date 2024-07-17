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

func sphere_tables(dump Dump, run_dir string, zero_lvl, c_x, c_y float32) {
	table_path := run_dir + "/sphere_ek.txt"
	table, err := os.Create(table_path)
	if err != nil {
		log.Fatalln("can't open file: ", table_path)
	}
	defer table.Close()

	table_str := make([][]string, len(dump.timesteps)+1)
	R_start := 10
	R_stop := 30
	d_R := 5
	for i, _ := range table_str {
		table_str[i] = make([]string, (R_stop-R_start)/d_R+2)
	}
	table_str[0][0] = "timestep\\R"

	for R := R_start; R <= R_stop; R += d_R {
		log.Println("table Ek - R:", R)
		j := (R-R_start)/d_R + 1
		table_str[0][j] = strconv.Itoa(R)
		for i, timestep := range dump.timesteps {
			var sum_ek float32
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
				}
			}
			table_str[i+1][j] = strconv.FormatFloat(float64(sum_ek), 'f', -1, 32)
			table_str[i+1][0] = strconv.Itoa(timestep)
		}
	}

	for _, row := range table_str {
		for j, cell := range row {
			if j > 0 {
				table.WriteString("\t")
			}
			table.WriteString(cell)
		}
		table.WriteString("\n")
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

	sphere_tables(dump, run_dir, zero_lvl, center_x, center_y)
}
