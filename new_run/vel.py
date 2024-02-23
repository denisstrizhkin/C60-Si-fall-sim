import math

energy = 8
vel = math.sqrt(energy * 1000) * 5.174

angle_1 = 16
angle_2 = 90 - angle_1

cos_1 = math.cos(math.radians(angle_1))
cos_2 = math.cos(math.radians(angle_2))

vel_z = vel * cos_1
vel_x = vel * cos_2

z = 150
lattice = 5.43

x0 = z * (vel_x / vel_z)

zlo = lattice * 6.5
zhi = zlo + lattice * 14

x_offset = lattice * 12

xlo = x_offset - x0 * (z - zlo) / z
xhi = x_offset - x0 * (z - zhi) / z

print(f"cos {angle_1}: {cos_1}")
print(f"cos {angle_2}: {cos_2}")
print("---")
print(f"vel: {vel}")
print("---")
print(f"vel z: {vel_z}")
print(f"vel x: {vel_x}")
print("---")
print(f"z: {z}")
print(f"xlo: {xlo}")
print(f"xhi: {xhi}")
