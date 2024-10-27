import math

BLOCK_OFFSET = 6
BLOCK_HEIGHT = 60

energy = 8
cluster_count = 60
cluster_amass = 12.011
vel = math.sqrt(energy * 1000 / cluster_amass / cluster_count) * 138.842

angle_1 = 8
angle_2 = 90 - angle_1

cos_1 = math.cos(math.radians(angle_1))
cos_2 = math.cos(math.radians(angle_2))

vel_z = vel * cos_1
vel_x = vel * cos_2

lattice = 5.43

z_offset = BLOCK_OFFSET * lattice
zlo = z_offset
zhi = zlo + BLOCK_HEIGHT * lattice
z = zhi + 10

print(f"variable block_zlo equal 'v_zero_lvl + {BLOCK_OFFSET - 0.1} * v_lattice'")
print(f"variable block_zhi equal 'v_block_zlo + {BLOCK_HEIGHT + 0.2} * v_lattice'")
print("---")

zlo += 170

x0 = z * (vel_x / vel_z)
x_offset = lattice * 12
xlo = x_offset - x0 * (z - zlo) / z
xhi = x_offset - x0 * (z - zhi) / z

print(f"variable cos1 equal '{cos_1}' # ({angle_1})")
print(f"variable cos2 equal '{cos_2}' # ({angle_2})")
print("---")
print(f"vel: {vel}")
print("---")
print(f"vel z: {vel_z}")
print(f"vel x: {vel_x}")
print("---")
print(f"z: {z}")
print(f"xlo = {xlo}")
print(f"xhi = {xhi}")
