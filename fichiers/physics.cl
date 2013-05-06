
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

void atom_collision_loop(int atom, __global float *pos, __global float *speed, int N, float coldist)
{
	int i;
	char colfound = 0;

	float3 mypos, otherpos;

	mypos.x = pos[atom];
	mypos.y = pos[atom + ROUND(N)];
	mypos.z = pos[atom + 2 * ROUND(N)];

	for (i = 0; i < atom; i++) {
		otherpos.x = pos[i];
		otherpos.y = pos[i + ROUND(N)];
		otherpos.z = pos[i + 2 * ROUND(N)];

		float d = distance(mypos, otherpos);

		if (d <= coldist) {
			colfound = 1;
			speed[i] = 0;
			speed[i + ROUND(N)] = 0;
			speed[i + 2 * ROUND(N)] = 0;
		}
	}		

	if (colfound == 1) {
		speed[atom] = 0;
		speed[atom + ROUND(N)] = 0;
		speed[atom + 2 * ROUND(N)] = 0;
	}
}


void atom_collision_v1(__global float *pos, __global float *speed, float radius)
{
	int N = get_global_size(0);
	int atom_no = get_global_id(0);

	atom_collision_loop(atom_no, pos, speed, N, 2*radius);
}


void atom_collision_v2(__global float *pos, __global float *speed, float radius)
	// pas forcément plus efficace car cette fois-ci nous avons deux fois
	// moins de threads
{
	int N = get_global_size(0);
	int atom1 = get_global_id(0);
	int atom2 = N - get_global_id(0) - 1;

	if (atom1 < N / 2) {
		atom_collision_loop(atom1, pos, speed, N, 2*radius);
		atom_collision_loop(atom2, pos, speed, N, 2*radius);
	}

	else if (atom1 == atom2) {
		atom_collision_loop(atom1, pos, speed, N, 2*radius);
	}
}

__kernel
void atom_collision(__global float *pos, __global float *speed, float radius)
{
	//atom_collision_v1(pos, speed, radius);
	atom_collision_v2(pos, speed, radius);
}

__kernel
void gravity(__global float *pos, __global float *speed, float g)
{
	int N = get_global_size(0);
	int atom = get_global_id(0);

	speed[atom + ROUND(N)] -= g;
}

__kernel
void lennard_jones(__global float *pos, __global float *speed, float radius)
{
}
