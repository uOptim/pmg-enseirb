
#pragma OPENCL EXTENSION all : enable

#define ALIGN     16
#define ROUND(n)  (((size_t)(n)+ALIGN-1)&(~(size_t)(ALIGN-1)))

// This kernel is executed with a very high number of threads = total number of vertices * 3
__kernel
void eating(__global float *vbo, float dy, unsigned vertices_per_atom)
{
	int index = get_global_id(0);
	int tpa = 3 * vertices_per_atom; // threads per atom
	// ghost or pacman?
	float enable = (index / tpa) % 2 ? 0.0f : 1.0f;
	// y coordinate?
	float is_y = ((index % 3) == 1) ? 1.0f : 0.0f;
	// vertex belongs to upper part of sphere?
	float is_up = (index % tpa >= tpa/2) ? 1.0f : -1.0f;

	vbo[index] += dy * is_y * is_up * enable;
}

__kernel
void move_vertices(__global float *vbo, __global float *speed, unsigned vertices_per_atom)
{
	int index = get_global_id(0);
	int tpa = 3 * vertices_per_atom; // threads per atom
	int atom_no = index / tpa;
	int no_atoms = get_global_size(0) / tpa;

	vbo[index] += speed[(index % 3) * ROUND(no_atoms) + atom_no];
}

__kernel
void update_position(__global float *pos, __global float *speed, unsigned N)
{
	int index = get_global_id(0);
	int atom_no = index % N;
	int compo = index / N;
	int true_idx = atom_no + compo * ROUND(N);
	pos[true_idx] += speed[true_idx];
}

__kernel
void border_collision(__global float *pos, __global float *speed, __constant float *min, __constant float *max, float radius, unsigned N)
{
	int index = get_global_id(0);
	int atom_no = index % N;
	int compo = index / N;

	int true_idx = atom_no + compo * ROUND(N);

	if (pos[true_idx] < (min[compo]+radius) || pos[true_idx] > (max[compo]-radius)) {
		speed[true_idx] *= -1;
	}
}

__kernel
void atom_collision(__global float *pos, __global float *speed, float radius)
{
}

__kernel
void gravity(__global float *pos, __global float *speed, float g)
{
}

__kernel
void lennard_jones(__global float *pos, __global float *speed, float radius)
{
}
