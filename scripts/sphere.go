package main

import (
	"bufio"
	"log"
	"math"
	"os"
	"strconv"
	"strings"
	"sync"
)

const (
	FLOAT_PREC = 5
)

type Dump struct {
	data      [][][]float64
	timesteps map[int]int
	keys      map[string]int
	path      string
}

func readAllLines(path string, max_timestep int) (lines []string, timesteps []int) {
	file, err := os.Open(path)
	if err != nil {
		log.Fatalf("opening dump file: %s - %v", path, err)
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	lines = make([]string, 0)
	timesteps = make([]int, 0)
	for i := 0; scanner.Scan(); i++ {
		if scanner.Text() == "ITEM: TIMESTEP" {
			if len(lines) > 0 {
				// start job
				lines = nil
			}
			lines = append(lines, scanner.Text())
			if !scanner.Scan() {
				break
			}
			lines = append(lines, scanner.Text())
			timestep := parse_int(scanner.Text())
			timesteps = append(timesteps, i)
			if timestep > max_timestep {
				return
			}
			i++
		}
		lines = append(lines, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		log.Fatalf("reading dump file: %s - %v", path, err)
	}
	timesteps = append(timesteps, len(lines))
	return
}

func dumpFromLines(wg *sync.WaitGroup, dump *Dump, lines []string) {
	defer wg.Done()
	dump.data = make([][][]float64, 0)
	dump.timesteps = make(map[int]int)
	dump.keys = make(map[string]int)
	atom_count := -1
	current_timestep := -1
	current_timestep_i := -1
	for i := 0; i < len(lines); i++ {
		if lines[i] == "ITEM: TIMESTEP" {
			i++
			current_timestep = parse_int(lines[i])
			current_timestep_i++
			dump.timesteps[current_timestep] = current_timestep_i
			log.Println("reading timestep:", current_timestep)
		} else if lines[i] == "ITEM: NUMBER OF ATOMS" {
			i++
			atom_count = parse_int(lines[i])
		} else if strings.Contains(lines[i], "ITEM: ATOMS") {
			if len(dump.keys) == 0 {
				for i, key := range strings.Split(lines[i], " ")[2:] {
					dump.keys[key] = i
					dump.data = append(dump.data, make([][]float64, 0))
				}
			}
			for _, i := range dump.keys {
				dump.data[i] = append(dump.data[i], make([]float64, atom_count))
			}
			i++
			for j := range atom_count {
				tokens := strings.Split(lines[i+j], " ")
				for _, i := range dump.keys {
					dump.data[i][current_timestep_i][j] = parse_float(tokens[i])
				}
			}
		}
	}
}

func NewDump(path string, max_timestep, threads int) (dump Dump) {
	lines, timesteps := readAllLines(path, max_timestep)
	finalI := len(timesteps) - 1
	chunkSize := (finalI) / threads
	dumps := make([]Dump, threads)
	wg := &sync.WaitGroup{}
	for i := range threads {
		start := i * chunkSize
		end := start + chunkSize
		if end < finalI && end+chunkSize > finalI {
			end = finalI
		}
		log.Printf(
			"processing chunk: [%d, %d] - [%d, %d]\n",
			start, end, timesteps[start], timesteps[end],
		)
		wg.Add(1)
		go dumpFromLines(wg, &dumps[i], lines[timesteps[start]:timesteps[end]])
	}
	wg.Wait()
	dump = dumps[0]
	for i := 1; i < len(dumps); i++ {
		for timestep, i := range dumps[i].timesteps {
			dump.timesteps[timestep] = len(dump.data[0]) + i
		}
		for _, j := range dump.keys {
			dump.data[j] = append(dump.data[j], dumps[i].data[j]...)
		}
	}
	return
}

func (d Dump) extract(field string, timestep int) []float64 {
	return d.data[d.keys[field]][d.timesteps[timestep]]
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
	zero_lvl := -math.MaxFloat64
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

func sphere_tables(dump Dump, zero_lvl, c_x, c_y float64) (tab_ek, tab_count [][]float64, rows, cols []string) {
	tab_ek = make([][]float64, len(dump.timesteps))
	tab_count = make([][]float64, len(dump.timesteps))
	rows = make([]string, len(dump.timesteps))
	var R_start float64 = 10
	var R_stop float64 = 30
	var R_d float64 = 5
	R_count := int((R_stop-R_start)/R_d + 1)
	for timestep, i := range dump.timesteps {
		tab_ek[i] = make([]float64, R_count)
		tab_count[i] = make([]float64, R_count)
		rows[i] = strconv.Itoa(timestep)
	}
	cols = make([]string, R_count)
	for j := range R_count {
		R := float64(j)*R_d + R_start
		cols[j] = strconv.FormatFloat(R, 'f', FLOAT_PREC, 64)
		for timestep, i := range dump.timesteps {
			x := dump.extract("x", timestep)
			y := dump.extract("y", timestep)
			z := dump.extract("z", timestep)
			ke := dump.extract("c_atom_ke", timestep)
			for a_i := range x {
				dx := x[a_i] - c_y
				dy := y[a_i] - c_x
				dz := z[a_i] - zero_lvl
				m := dx*dx + dy*dy + dz*dz
				if m <= R*R && z[a_i] <= zero_lvl+1 {
					tab_ek[i][j] += ke[a_i]
					tab_count[i][j]++
				}
			}
		}
	}
	return
}

func histogram(vals []float64, start, end float64, count int, cut_ends bool) (bins, counts []float64) {
	bins = make([]float64, count)
	counts = make([]float64, count)
	length := end - start
	width := length / float64(count)

	for i := range count {
		bins[i] = start + width*float64(i) + width/2
	}

	for _, val := range vals {
		if val < start {
			if cut_ends {
				continue
			}
			val = start
		}
		if val > end {
			if cut_ends {
				continue
			}
			val = end
		}
		index := int((val - start) / width)
		if index == count {
			index--
		}
		counts[index]++
	}

	return bins, counts
}

func velocity_table(dump Dump, zero_lvl, c_x, c_y float64) (tab [][]float64, rows, cols []string) {
	tab = make([][]float64, len(dump.timesteps))
	angles := make([][]float64, len(dump.timesteps))
	rows = make([]string, len(dump.timesteps))
	var cyllinder_r float64 = 20
	var cyllinder_h float64 = 10
	var bin_width float64 = 10
	var bin_start float64 = -180
	var bin_end float64 = 180
	bin_count := int((bin_end - bin_start) / bin_width)
	cols = make([]string, bin_count)
	for timestep, j := range dump.timesteps {
		rows[j] = strconv.Itoa(timestep)
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
				angles[j] = append(angles[j], angle)
			}
		}
	}
	for j := range angles {
		bins, counts := histogram(angles[j], bin_start, bin_end, bin_count, true)
		tab[j] = counts
		if j == 0 {
			for i, bin := range bins {
				cols[i] = strconv.FormatFloat(bin, 'f', FLOAT_PREC, 64)
			}
		}
	}
	return
}

func energy_distr(dump Dump, zero_lvl, c_x, c_y float64) (tab [][]float64, rows, cols []string) {
	tab = make([][]float64, len(dump.timesteps))
	ke := make([][]float64, len(dump.timesteps))
	rows = make([]string, len(dump.timesteps))
	var cyllinder_r float64 = 20
	var cyllinder_h float64 = 10
	var bin_width float64 = 0.025
	var bin_start float64 = 0
	var bin_end float64 = 2
	bin_count := int((bin_end - bin_start) / bin_width)
	cols = make([]string, bin_count)
	for timestep, j := range dump.timesteps {
		rows[j] = strconv.Itoa(timestep)
		x := dump.extract("x", timestep)
		y := dump.extract("y", timestep)
		z := dump.extract("z", timestep)
		kin_e := dump.extract("c_atom_ke", timestep)
		for i := range x {
			dx := x[i] - c_x
			dy := y[i] - c_y
			dz := z[i] - zero_lvl
			m := dx*dx + dy*dy
			if m <= cyllinder_r*cyllinder_r && math.Abs(dz) <= cyllinder_h {
				ke[j] = append(ke[j], kin_e[i])
			}
		}
	}
	for j := range ke {
		bins, counts := histogram(ke[j], bin_start, bin_end, bin_count, false)
		tab[j] = counts
		if j == 0 {
			for i, bin := range bins {
				cols[i] = strconv.FormatFloat(bin, 'f', FLOAT_PREC, 64)
			}
		}
	}
	return
}

func write_table(vals [][]float64, rows, cols []string, corner, path string) {
	f, err := os.Create(path)
	if err != nil {
		log.Fatalf("can't open file: %s - %v", path, err)
	}
	defer f.Close()

	w := bufio.NewWriter(f)
	defer w.Flush()
	w.WriteString(corner)
	for _, col := range cols {
		w.WriteRune('\t')
		w.WriteString(col)
	}
	w.WriteRune('\n')

	for i, row := range rows {
		w.WriteString(row)
		for _, val := range vals[i] {
			w.WriteRune('\t')
			w.WriteString(strconv.FormatFloat(val, 'f', FLOAT_PREC, 64))
		}
		w.WriteRune('\n')
	}
}

func create_sphere_tables(dump Dump, run_dir string, zero_lvl, center_x, center_y float64) {
	ke_path := run_dir + "/sphere_ke.txt"
	count_path := run_dir + "/sphere_count.txt"
	tab_ke, tab_count, rows, cols := sphere_tables(dump, zero_lvl, center_x, center_y)
	corner := "timestep\\R"
	write_table(tab_ke, rows, cols, corner, ke_path)
	write_table(tab_count, rows, cols, corner, count_path)
}

func create_vel_dist_table(dump Dump, run_dir string, zero_lvl, center_x, center_y float64) {
	path := run_dir + "/vel_distrib.txt"
	tab, rows, cols := velocity_table(dump, zero_lvl, center_x, center_y)
	corner := "timestep\\deg"
	write_table(tab, rows, cols, corner, path)
}

func create_energy_dist_table(dump Dump, run_dir string, zero_lvl, center_x, center_y float64) {
	path := run_dir + "/ke_distrib.txt"
	tab, rows, cols := energy_distr(dump, zero_lvl, center_x, center_y)
	corner := "timestep\\deg"
	write_table(tab, rows, cols, corner, path)
}

func main() {
	args := os.Args[1:]

	if len(args) != 1 {
		log.Fatal("usage: sphere.go RUN_DIR")
	}
	run_dir := args[0]
	dump_during := run_dir + "/dump.during"
	log.Println("using run_dir:", run_dir)

	dump := NewDump(dump_during, 5100, 4)
	log.Println("dump len: ", len(dump.data[0]))
	zero_lvl := calc_zero_lvl(dump, 0)
	center_x, center_y := calc_cluster_center(dump, 0)
	log.Printf("zero_lvl: %f, center: (%f, %f)\n", zero_lvl, center_x, center_y)

	create_sphere_tables(dump, run_dir, zero_lvl, center_x, center_y)
	create_vel_dist_table(dump, run_dir, zero_lvl, center_x, center_y)
	create_energy_dist_table(dump, run_dir, zero_lvl, center_x, center_y)
}
