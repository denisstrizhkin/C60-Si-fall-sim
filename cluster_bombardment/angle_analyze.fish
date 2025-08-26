#!/usr/bin/env fish

set incident 0 10 20 30 40 50 60 70 80
# set incident 10
set azimuthal 0 12 24 36 45
# set azimuthal 0

function rim
    set -l d $argv[1]
    rim-analysis multi -t (nproc) $d >$d/rim_table.txt
end

function crater
    set -l d $argv[1]
    crater-analysis multi -t (nproc) $d >$d/crater_table.txt
end

function surface
    set -l d $argv[1]
    surface-analysis -z 82.7813 multi -t (nproc) $d
end

function collect_surface
    set -l d $argv[1]
    set -l out $argv[2]
    set -l d_b (basename $d)
    mkdir -p $out
    cp $d/surface_2d.png $out/{$d_b}.png
    cp $d/surface_coords.txt $out/{$d_b}.txt
end

function collect_txt
    set -l d $argv[1]
    set -l out $argv[2]
    set -l d_b (basename $d)
    set -l out $out/$d_b
    mkdir -p $out
    rsync $d/crater_table.txt \
        $d/carbon_dist_parsed.txt \
        $d/carbon_table.txt \
        $d/clusters_table_parsed_energy_dist.txt \
        $d/clusters_table_parsed_number_dist.txt \
        $d/clusters_table_parsed_sum.txt \
        $d/clusters_table_parsed.txt \
        $d/surface_table.txt \
        $d/surface_coords.txt \
        $d/surface_2d.png \
        $d/rim_table.txt \
        $out
end

function run
    set -l cmd $argv[1]
    set -l args $argv[2..]
    for a1 in $incident
        for a2 in $azimuthal
            set -l d results/0K_8keV_angles_{$a1}_{$a2}
            echo -n "Staring - $cmd: $d: "
            $cmd $d $args
            and echo done
        end
    end
end

switch $argv[1]
    case rim
        run rim
    case crater
        run crater
    case surface
        run surface
    case collect_surface
        set -l out (basename (pwd))_surface
        run collect_surface $out
        tar -cJf $out.tar.xz $out
    case collect_txt
        set -l out (basename (pwd))_txt
        run collect_txt $out
        tar -cJf $out.tar.xz $out
    case "*"
        echo "Unknown cmd: $argv[1]"
end
