import math

energy = 8
vel = math.sqrt(energy * 1000) * 5.174

cos_8 = math.cos(math.radians(8))
cos_82 = math.cos(math.radians(82))

vel_z = vel * cos_8
vel_x = vel * cos_82

z = 100
x = z * (vel_x / vel_z)

print(f"cos 8: {cos_8}")
print(f"cos 82: {cos_82}")
print("---")
print(f"vel: {vel}")
print("---")
print(f"vel z: {vel_z}")
print(f"vel x: {vel_x}")
print("---")
print(f"z: {z}")
print(f"x: {x}")
