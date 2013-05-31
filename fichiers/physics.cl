#pragma OPENCL EXTENSION all : enable

#define ALIGN     16
#define ROUND(n)  (((size_t)(n)+ALIGN-1)&(~(size_t)(ALIGN-1)))

#define SIGMA 1.25
#define EPSILON 0.005

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

	if (pos[true_idx] < (min[compo]+radius) && speed[true_idx] < 0) {
		speed[true_idx] *= -1;
	}
	
	if (pos[true_idx] > (max[compo]-radius) && speed[true_idx] > 0) {
		speed[true_idx] *= -1;
	}
}


void atom_collision_loop(int atom, __global float *pos, __global float *speed, int N, float coldist)
{
	float3 Ca, Cb;
	float3 i, j, k;

	float3 x, y, z;
	x.x = 1; x.y = 0; x.z = 0;
	y.x = 0; y.y = 1; y.z = 0;
	z.x = 0; z.y = 0; z.z = 1;

	float3 Va, Vb;
	float3 m1, m2, m3;    // M
	float3 mt1, mt2, mt3; // tM
	float3 Var, Vpar;     // V_A_r, V'_A_r
	float3 Vbr, Vpbr;     // V_B_r, V'_B_r

	Ca.x = pos[atom];
	Ca.y = pos[atom + ROUND(N)];
	Ca.z = pos[atom + 2 * ROUND(N)];

	Va.x = speed[atom];
	Va.y = speed[atom + ROUND(N)];
	Va.z = speed[atom + 2 * ROUND(N)];

	int a;
	for (a = 0; a < atom; a++) {
		Cb.x = pos[a];
		Cb.y = pos[a + ROUND(N)];
		Cb.z = pos[a + 2 * ROUND(N)];

		if (distance(Ca, Cb) <= coldist) {
			Vb.x = speed[a];
			Vb.y = speed[a + ROUND(N)];
			Vb.z = speed[a + 2 * ROUND(N)];

			i = normalize(Ca - Cb);
			if (i.x == -1) i.x = 1;// y and j are 0 if x is 1/-1
			j = normalize(cross(x, i));
			k = cross(i, j);

			m1 = i;
			m2.x = -i.y; m2.y = (i.x + (i.z * i.z)/(1 + i.x)); m2.z = (-i.y * i.z)/(1 + i.x);
			m3.x = -i.z; m3.y = (-i.y * i.z)/(1 + i.x);        m3.z = (i.x + (i.y * i.y)/(1 + i.x));

			mt1.x = m1.x; mt1.y = m2.x; mt1.z = m3.x;
			mt2.x = m1.y; mt2.y = m2.y; mt2.z = m3.y;
			mt3.x = m1.z; mt3.y = m2.z; mt3.z = m3.z;

			Var.x = dot(Va, m1); Var.y = dot(Va, m2); Var.z = dot(Va, m3);
			Vbr.x = dot(Vb, m1); Vbr.y = dot(Vb, m2); Vbr.z = dot(Vb, m3);

			Vpar = Var; Vpar.x = Vbr.x;
			Vpbr = Vbr; Vpbr.x = Var.x;

			speed[atom]              = dot(Vpar, mt1);
			speed[atom + ROUND(N)]   = dot(Vpar, mt2);
			speed[atom + 2*ROUND(N)] = dot(Vpar, mt3);

			speed[a]              = dot(Vpbr, mt1);
			speed[a + ROUND(N)]   = dot(Vpbr, mt2);
			speed[a + 2*ROUND(N)] = dot(Vpbr, mt3);
		}
	}
}

void atom_collision_loop_sync(int atom, __global float *pos, __global float *speed, int N, float coldist);

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


void atom_collision_v3(__global float *pos, __global float *speed, float radius, int N)
{
	__local float3 colone[16];
	__local float3 colone_speed[16];

	int local_id = get_local_id(0);
	int global_id = get_group_id(0);

	int u = (int)sqrt(2.0*(global_id+1));

	if (u * (u+1) / 2 < global_id+1) {
		u = u + 1;
	}

	int a = 16*(u - 1);
	int b = 16*(global_id - (u-1)*u/2);

	// fill buffer
	if (b+local_id < N) {
		colone[local_id].x = pos[b + local_id];
		colone[local_id].y = pos[b + local_id + ROUND(N)];
		colone[local_id].z = pos[b + local_id + 2*ROUND(N)];
		colone_speed[local_id].x = speed[b + local_id];
		colone_speed[local_id].y = speed[b + local_id + ROUND(N)];
		colone_speed[local_id].z = speed[b + local_id + 2*ROUND(N)];
	}

	barrier(CLK_LOCAL_MEM_FENCE);


	float3 Ca, Cb, Va, Vb;
	Ca.x = pos[a + local_id];
	Ca.y = pos[a + local_id + ROUND(N)];
	Ca.z = pos[a + local_id + 2*ROUND(N)];
	Va.x = speed[a + local_id];
	Va.y = speed[a + local_id + ROUND(N)];
	Va.z = speed[a + local_id + 2*ROUND(N)];

	float3 x, y, z;
	x.x = 1; x.y = 0; x.z = 0;
	y.x = 0; y.y = 1; y.z = 0;
	z.x = 0; z.y = 0; z.z = 1;

	float3 m1, m2, m3;    // M
	float3 mt1, mt2, mt3; // tM
	float3 Var, Vpar;     // V_A_r, V'_A_r
	float3 Vbr, Vpbr;     // V_B_r, V'_B_r

	int t;
	float3 i, j, k;
	for (t = 0; t < 16; t++) {

		if (a == b && t == local_id) {
			break;
		}

		Cb = colone[t];
		Vb = colone_speed[t];

		if (distance(Ca, Cb) <= 2*radius) {

			i = normalize(Ca - Cb);

			if (i.x == -1) {
				i.x = 1; // y and j are 0 if x is 1/-1
			}

			j = normalize(cross(x, i));
			k = cross(i, j);

			m1 = i;
			m2.x = -i.y; m2.y = (i.x + (i.z * i.z)/(1 + i.x)); m2.z = (-i.y * i.z)/(1 + i.x);
			m3.x = -i.z; m3.y = (-i.y * i.z)/(1 + i.x) ; m3.z = (i.x + (i.y * i.y)/(1 + i.x));

			mt1.x = m1.x; mt1.y = m2.x; mt1.z = m3.x;
			mt2.x = m1.y; mt2.y = m2.y; mt2.z = m3.y;
			mt3.x = m1.z; mt3.y = m2.z; mt3.z = m3.z;

			Var.x = dot(Va, m1); Var.y = dot(Va, m2); Var.z = dot(Va, m3);
			Vbr.x = dot(Vb, m1); Vbr.y = dot(Vb, m2); Vbr.z = dot(Vb, m3);

			Vpar = Var; Vpar.x = Vbr.x;
			Vpbr = Vbr; Vpbr.x = Var.x;

			speed[a + local_id]              = dot(Vpar, mt1);
			speed[a + local_id + ROUND(N)]   = dot(Vpar, mt2);
			speed[a + local_id + 2*ROUND(N)] = dot(Vpar, mt3);

			speed[b + t]              = dot(Vpbr, mt1);
			speed[b + t + ROUND(N)]   = dot(Vpbr, mt2);
			speed[b + t + 2*ROUND(N)] = dot(Vpbr, mt3);
		}
	}
}

__kernel
void atom_collision(__global float *pos, __global float *speed, float radius, int N)
{
	//atom_collision_v1(pos, speed, radius);
	atom_collision_v2(pos, speed, radius);
	//atom_collision_v3(pos, speed, radius, N);
}

__kernel
void gravity(__global float *pos, __global float *speed, float g)
{
	int N = get_global_size(0);
	int atom = get_global_id(0);

	speed[atom + ROUND(N)] -= g;
}

void lennard_jones_v1(__global float *pos, __global float *speed, float radius)
{
	int N = get_global_size(0);
	int atom = get_global_id(0);

	float d;
	float3 Ca, Cb, diff;

	Ca.x = pos[atom];
	Ca.y = pos[atom + ROUND(N)];
	Ca.z = pos[atom + 2 * ROUND(N)];

	float3 speedDelta;

	speedDelta.x = 0;
	speedDelta.y = 0;
	speedDelta.z = 0;

	int a;
	for (a = 0; a < N; a++) {
		if (a == atom)
			continue;
		Cb.x = pos[a];
		Cb.y = pos[a + ROUND(N)];
		Cb.z = pos[a + 2 * ROUND(N)];

		d = distance(Ca, Cb);

		float tmp = SIGMA / d;

		float energy = 4 * EPSILON * (pow(tmp,12) - pow(tmp,6));

		diff = normalize(Ca - Cb);

		speedDelta.x += diff.x * energy;
		speedDelta.y += diff.y * energy;
		speedDelta.z += diff.z * energy;
	}

	speed[atom] += speedDelta.x;
	speed[atom + ROUND(N)] += speedDelta.y;
	speed[atom + 2 * ROUND(N)] += speedDelta.z;
}

#define SLICE_SIZE 16

void lennard_jones_v2(__global float *pos, __global float *speed, float radius, int nb_atoms)
{

	__local float3 next_atoms[SLICE_SIZE];

	int group_id = get_group_id(0);
	int global_id = get_global_id(0);
	int local_id = get_local_id(0);
	int nb_slices = (nb_atoms - 1) / 16 + 1;

	float3 my_atom;
	if (global_id < nb_atoms){
		my_atom.x = pos[global_id];
		my_atom.y = pos[global_id + ROUND(nb_atoms)];
		my_atom.z = pos[global_id + 2 * ROUND(nb_atoms)];
	}

	float3 speedDelta;

	speedDelta.x = 0;
	speedDelta.y = 0;
	speedDelta.z = 0;

	int slice_no, j;
	for (slice_no = 0; slice_no < nb_slices; slice_no++) {

		// Loading an atom for workgroup
		int to_load_id = slice_no * SLICE_SIZE + local_id;
		if (to_load_id < nb_atoms){
			next_atoms[local_id].x = pos[to_load_id];
			next_atoms[local_id].y = pos[to_load_id + ROUND(nb_atoms)];
			next_atoms[local_id].z = pos[to_load_id + 2 * ROUND(nb_atoms)];
		}

		// Waiting all threads to load
		barrier(CLK_LOCAL_MEM_FENCE);

		// Computing speed deltas for this slice
		if (global_id < nb_atoms){
			// Computing speed deltas
			for (j = 0; j < SLICE_SIZE; j++){
				int opposite_atom_id = slice_no * SLICE_SIZE + j;
				// not computing with non-existing atoms
				if (opposite_atom_id >= nb_atoms)
					break;
				//not applying force on himself
				if (opposite_atom_id == global_id)
					continue;

				float d = distance(my_atom, next_atoms[j]);

				float tmp = SIGMA / d;

				float energy = 4 * EPSILON * (pow(tmp,12) - pow(tmp,6));

				if (isnan(energy))
					speed[global_id + 2 * ROUND(nb_atoms)] = tmp;

				float3 diff = normalize(my_atom - next_atoms[j]);

				speedDelta.x += diff.x * energy;
				speedDelta.y += diff.y * energy;
				speedDelta.z += diff.z * energy;
			}
		}

		// Waiting all threads before starting next slice
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Applying speed_deltats
	if (global_id < nb_atoms && !isnan(speedDelta.x)){
		speed[global_id] += speedDelta.x;
		speed[global_id + ROUND(nb_atoms)] += speedDelta.y;
		speed[global_id + 2 * ROUND(nb_atoms)] += speedDelta.z;
	}
}

#define LENNARD_JONES_VERSION 2

__kernel
void lennard_jones(__global float *pos, __global float *speed, float radius, int nb_atoms){
#if LENNARD_JONES_VERSION == 1
  lennard_jones_v1(pos, speed, radius);
#endif
#if LENNARD_JONES_VERSION == 2
  lennard_jones_v2(pos, speed, radius, nb_atoms);
#endif
}

