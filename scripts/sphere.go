package main

import (
	"bufio"
	"fmt"
	"log"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
)

type Atom struct {
	type_       int
	x, y, z, ek float32
}

type Dump struct {
	timesteps map[int][]Atom
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

func read_dump(dump_path string) Dump {
	file, err := os.Open(dump_path)
	if err != nil {
		log.Fatalln("reading dump file: ", dump_path, err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	dump := Dump{
		timesteps: make(map[int][]Atom),
	}

	is_read_atom_count := false
	is_read_atoms := false
	is_read_timestep := false

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
			atom_count := parse_int(scanner.Text())
			dump.timesteps[current_timestep] = make([]Atom, atom_count)
			is_read_atom_count = false
			continue
		}

		if strings.Contains(scanner.Text(), "ITEM: ATOMS") {
			is_read_atoms = true
			continue
		}

		if is_read_atoms {
			// id type x y z vx vy vz c_atom_ke
			tokens := strings.Split(scanner.Text(), " ")
			current_atom++
			dump.timesteps[current_timestep][current_atom].type_ = parse_int(tokens[1])
			dump.timesteps[current_timestep][current_atom].x = parse_float(tokens[2])
			dump.timesteps[current_timestep][current_atom].y = parse_float(tokens[3])
			dump.timesteps[current_timestep][current_atom].z = parse_float(tokens[4])
			dump.timesteps[current_timestep][current_atom].ek = parse_float(tokens[8])
		}
	}

	if err := scanner.Err(); err != nil {
		fmt.Fprintln(os.Stderr, "reading standard input:", err)
	}

	return dump
}

func calc_zero_lvl(dump Dump, timestep int) float32 {
	var zero_lvl float32 = math.SmallestNonzeroFloat32
	for _, atom := range dump.timesteps[timestep] {
		if atom.z > zero_lvl && atom.type_ == 1 {
			zero_lvl = atom.z
		}
	}
	return zero_lvl
}

func calc_cluster_center(dump Dump, timestep int) (float32, float32) {
	var x, y float32 = 0, 0
	count := 0
	for _, atom := range dump.timesteps[timestep] {
		if atom.type_ == 2 {
			x += atom.x
			y += atom.y
			count++
		}
	}
	return x / float32(count), y / float32(count)
}

func sphere_tables(dump Dump, run_dir string, zero_lvl float32, x float32, y float32) {
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

	timesteps := make([]int, 0)
	for timestep := range dump.timesteps {
		timesteps = append(timesteps, timestep)
	}
	sort.Ints(timesteps)

	for R := R_start; R <= R_stop; R += d_R {
		log.Println("table Ek - R:", R)
		j := (R-R_start)/d_R + 1
		table_str[0][j] = strconv.Itoa(R)
		for i, timestep := range timesteps {
			var ek float32 = 0
			for _, atom := range dump.timesteps[timestep] {
				dx := atom.x - y
				dy := atom.y - x
				dz := atom.z - zero_lvl
				m := dx*dx + dy*dy + dz*dz
				if m <= float32(R*R) && atom.z <= zero_lvl+1 {
					ek += atom.ek
				}
			}
			table_str[i+1][j] = strconv.FormatFloat(float64(ek), 'f', -1, 32)
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

	dump := read_dump(dump_during)
	zero_lvl := calc_zero_lvl(dump, 0)
	center_x, center_y := calc_cluster_center(dump, 0)
	log.Printf("zero_lvl: %f, center: (%f, %f)\n", zero_lvl, center_x, center_y)

	sphere_tables(dump, run_dir, zero_lvl, center_x, center_y)
}
