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
	int a;

	float d;
	float3 i, j, k;
	float3 Ca, Cb;

	Ca.x = pos[atom];
	Ca.y = pos[atom + ROUND(N)];
	Ca.z = pos[atom + 2 * ROUND(N)];

	float3 x, y, z;
	x.x = 1; x.y = 0; x.z = 0;
	y.x = 0; y.y = 1; y.z = 0;
	z.x = 0; z.y = 0; z.z = 1;

	float3 Va, Vb;
	float3 m1, m2, m3;    // M
	float3 mt1, mt2, mt3; // tM
	float3 Var, Vpar;     // V_A_r, V'_A_r
	float3 Vbr, Vpbr;     // V_B_r, V'_B_r

	Va.x = speed[atom];
	Va.y = speed[atom + ROUND(N)];
	Va.z = speed[atom + 2 * ROUND(N)];

	for (a = 0; a < atom; a++) {
		Cb.x = pos[a];
		Cb.y = pos[a + ROUND(N)];
		Cb.z = pos[a + 2 * ROUND(N)];

		d = distance(Ca, Cb);

		if (d <= coldist) {
			Vb.x = speed[a];
			Vb.y = speed[a + ROUND(N)];
			Vb.z = speed[a + 2 * ROUND(N)];

			i = normalize(Ca - Cb);
			if (i.x == -1) i.x = 1;// y and j are 0 if x is 1/-1
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
	int N = get_global_size(0);
	int atom = get_global_id(0);

	float d, energy;
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

	barrier(CLK_LOCAL_MEM_FENCE);

	speed[atom] += speedDelta.x;
	speed[atom + ROUND(N)] += speedDelta.y;
	speed[atom + 2 * ROUND(N)] += speedDelta.z;
}

// Work in progress
void atom_collision_loop_sync(int atom, __global float *pos, __global float *speed, int N, float coldist)
{
	int a;

	float d;
	float3 i, j, k;
	float3 Ca, Cb;

	Ca.x = pos[atom];
	Ca.y = pos[atom + ROUND(N)];
	Ca.z = pos[atom + 2 * ROUND(N)];

	float3 x, y, z;
	x.x = 1; x.y = 0; x.z = 0;
	y.x = 0; y.y = 1; y.z = 0;
	z.x = 0; z.y = 0; z.z = 1;

	float3 Va, Vb, deltaV;
	float3 m1, m2, m3;    // M
	float3 mt1, mt2, mt3; // tM
	float3 Var, Vpar;     // V_A_r, V'_A_r
	float3 Vbr, Vpbr;     // V_B_r, V'_B_r

	Va.x = speed[atom];
	Va.y = speed[atom + ROUND(N)];
	Va.z = speed[atom + 2 * ROUND(N)];

	deltaV.x += 0;
	deltaV.y += 0;
	deltaV.z += 0;

	for (a = 0; a < N; a++) {
		if (a == atom)
			continue;
		Cb.x = pos[a];
		Cb.y = pos[a + ROUND(N)];
		Cb.z = pos[a + 2 * ROUND(N)];

		d = distance(Ca, Cb);

		if (d <= coldist) {
			Vb.x = speed[a];
			Vb.y = speed[a + ROUND(N)];
			Vb.z = speed[a + 2 * ROUND(N)];

			i = normalize(Ca - Cb);
			if (i.x == -1) i.x = 1;// y and j are 0 if x is 1/-1
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

			deltaV.x += dot(Vpar, mt1) - Va.x;
			deltaV.y += dot(Vpar, mt2) - Va.y;
			deltaV.z += dot(Vpar, mt3) - Va.z;
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	speed[atom]              += deltaV.x;
	speed[atom + ROUND(N)]   += deltaV.y;
	speed[atom + 2*ROUND(N)] += deltaV.z;
}
