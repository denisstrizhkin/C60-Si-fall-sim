# Creating Big Crystals

## General Commands for Crystal Heating

Use GPU suffix

```
package gpu 0
suffix gpu
```

Setup basic simulation parameters and lattice

```
units metal
dimension 3
boundary p p f
atom_style atomic
atom_modify map yes

lattice diamond 5.43 orient x 1 0 0 orient y 0 1 0 orient z 0 0 1
```

Setting up simulation box of a given length

```
region symbox block -${width} ${width} -${width} ${width} -${depth} ${depth} units lattice
create_box 1 symbox
```

Creating regions
```
region si_all block EDGE EDGE EDGE EDGE EDGE ${top}          units lattice
region si_fix block EDGE EDGE EDGE EDGE EDGE $(-v_depth+0.5) units lattice

region top block EDGE EDGE EDGE EDGE $(v_top -0.2) ${top} units lattice
```

Creating atoms
```
create_atoms 1 region si_all
mass 1 28.08553
```

Potentials
```
pair_style tersoff/zbl
pair_coeff * * potentials/SiC.tersoff.zbl Si
neighbor 13.0 bin
```

Groups
```
group si_all type 1
group si_fix region si_fix

group top region top

group nve subtract si_all si_fix
```

Vacancies (Voronoi)
```
compute voro si_all voronoi/atom occupation only_group

variable is_vac atom "c_voro[1]==0"
compute vac_sum si_all reduce sum v_is_vac
variable vacs equal "c_vac_sum"
```

Output variables: average Z coodrinate of top atom layer and simulation step
```
compute top_sum top reduce sum z
variable top_avrg equal "c_top_sum / count(top)"

variable step equal "step"
```

Minimize system
```
minimize 1e-20 1e-20 1000 1000
```

Give atoms minimal starting speed
```
velocity nve create ${start_temp} 4928459 dist gaussian
```

Fixes: nve thermostat (for heating) and average Z coordinate output
```
fix extra1 all print 10 "${step} ${top_avrg}" file graph1.txt screen no
fix f_norm nve nvt temp ${start_temp} ${end_temp} $(v_tdamp*dt)
```

Timestep setup, information output and simulation run
```
reset_timestep 0
timestep 0.001

thermo 50
thermo_style custom step pe ke etotal temp v_vacs

run ${heat_time}
```

Run plain nve to see final oscilations
```
unfix f_norm
fix f_nve nve nve
fix extra2 all print 10 "${step} ${top_avrg}" file graph2.txt screen no

run 20000
```

Save final crystal
```
write_data output.data
```

## Lateral duplication

Setting up simulation variable
```
variable xl equal "-v_width"
variable xr equal "v_width"

variable yl equal "-v_width"
variable yr equal "v_width"

variable zd equal "-v_depth"

variable lattice equal "5.43"

variable xy_offset equal "v_width * 2.0 * v_lattice"
```

Creating simulation box
```
lattice diamond ${lattice} orient x 1 0 0 orient y 0 1 0 orient z 0 0 1

region r_symbox block ${xl} ${xr} ${yl} ${yr} ${zd} 1000.0 units lattice
create_box 1 r_symbox
```

### Duplication

2x2
```
change_box all x scale 2.0 y scale 2.0

read_data input.data add merge  shift $(v_xy_offset * 0.5)  $(v_xy_offset * 0.5)  0.0
read_data input.data add append shift $(v_xy_offset * -0.5) $(v_xy_offset * 0.5)  0.0
read_data input.data add append shift $(v_xy_offset * 0.5)  $(v_xy_offset * -0.5) 0.0
read_data input.data add append shift $(v_xy_offset * -0.5) $(v_xy_offset * -0.5) 0.0
```

3x3
```
change_box all x scale 3.0 y scale 3.0

read_data input.data add merge
read_data input.data add append shift +${xy_offset} +${xy_offset} 0.0
read_data input.data add append shift -${xy_offset} +${xy_offset} 0.0
read_data input.data add append shift -${xy_offset} -${xy_offset} 0.0
read_data input.data add append shift +${xy_offset} -${xy_offset} 0.0
read_data input.data add append shift 0.0           +${xy_offset} 0.0
read_data input.data add append shift 0.0           -${xy_offset} 0.0
read_data input.data add append shift +${xy_offset} 0.0           0.0
read_data input.data add append shift -${xy_offset} 0.0           0.0
```

5x5
```
change_box all x scale 5.0 y scale 5.0

read_data input.data add merge
read_data input.data add append shift +${xy_offset} +${xy_offset} 0.0
read_data input.data add append shift -${xy_offset} +${xy_offset} 0.0
read_data input.data add append shift -${xy_offset} -${xy_offset} 0.0
read_data input.data add append shift +${xy_offset} -${xy_offset} 0.0
read_data input.data add append shift 0.0           +${xy_offset} 0.0
read_data input.data add append shift 0.0           -${xy_offset} 0.0
read_data input.data add append shift +${xy_offset} 0.0           0.0
read_data input.data add append shift -${xy_offset} 0.0           0.0

read_data input.data add append shift $(v_xy_offset * 2) $(v_xy_offset * 2) 0.0
read_data input.data add append shift $(v_xy_offset * 2) $(v_xy_offset * 1) 0.0
read_data input.data add append shift $(v_xy_offset * 2) $(v_xy_offset * 0) 0.0
read_data input.data add append shift $(v_xy_offset * 2) $(v_xy_offset * -1) 0.0
read_data input.data add append shift $(v_xy_offset * 2) $(v_xy_offset * -2) 0.0

read_data input.data add append shift $(v_xy_offset * -2) $(v_xy_offset * 2) 0.0
read_data input.data add append shift $(v_xy_offset * -2) $(v_xy_offset * 1) 0.0
read_data input.data add append shift $(v_xy_offset * -2) $(v_xy_offset * 0) 0.0
read_data input.data add append shift $(v_xy_offset * -2) $(v_xy_offset * -1) 0.0
read_data input.data add append shift $(v_xy_offset * -2) $(v_xy_offset * -2) 0.0

read_data input.data add append shift $(v_xy_offset * -1) $(v_xy_offset * 2) 0.0
read_data input.data add append shift $(v_xy_offset * 0) $(v_xy_offset * 2) 0.0
read_data input.data add append shift $(v_xy_offset * 1) $(v_xy_offset * 2) 0.0

read_data input.data add append shift $(v_xy_offset * -1) $(v_xy_offset * -2) 0.0
read_data input.data add append shift $(v_xy_offset * 0) $(v_xy_offset * -2) 0.0
read_data input.data add append shift $(v_xy_offset * 1) $(v_xy_offset * -2) 0.0
```

Save output
```
write_data output.data
```

## 80x80x96

### 0K

```
variable width equal "4"
variable depth equal "80"

variable top equal "15.3"

variable start_temp equal "1e-6"
variable end_temp   equal "1e-6"

variable tdamp equal "100"
variable heat_time equal "1 * 1000 * 1000"
```

### Heated (1000k)

## 160x160x32

## 160x160x96
