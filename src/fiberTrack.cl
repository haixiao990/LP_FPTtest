/** by Mo Xu, Cong Gao, Haixiao Du
** fiber tracking OpenCL kernel
** Each thread handles 1 seed from a sample. All threads traverse the entire cube from a sample. 
** To address many samples: 1) many kernels (may be too light burden for a thread), 2) iteration of one thread (need many images) or plus many work-groups(considering parallelism).
** Pro: requires less images simultanously(if use 1) to address many sample); read global memory more consecutively (plausible advantage); doesn't use LDS, so higher occupancy.
** Con: writes global memory intensitively and with bad pattern (trival if compute-bound); must use global atomic write if the number of points is small (considering parallelism)
**
** Another solution:
** Each work-group (256 threads) handles 1 seed from 256 samples, and iteratively handles 1 seed from all samples. All work-groups traverse the cube. 
** Pro: writes global memory with good pattern (use LDS to store the result); parallelism is high. Loads of threads within one quater-wavefront are more balanced.
** Con: writes LDS using atomic operation; needs 10KB LDS for one workgroup (if 10K nodes, use char to store); the occupancy is limited (trival if compute-bound); char atomic is not directly supported.
**		needs many images at one time.
**
** Disadvantage: very time-consuming when you just want to obtain probtrack from small number of seeds.
**
*/

//inline float distance2(float4 x, float4 y)
//{
//	float4 a = x - y;
//	return (a.x * a.x + a.y * a.y + a.z * a.z);
//}
//#define USE_F_THRESHOLD_AS_TERMINATE_CONDITION
//#define RECORD_INTEGER_POINT_ONLY
#define PI 3.141592653589793f
#define LOCAL_SIZE 256
#define f_threshold 0.05
#define ROUND_MAX 5

__kernel void fiberTrack(
	// output
	__global float4 * result,
	//__global int * result_step,
	__global int * result_iteration,
	//__global float * result_length,
	__global int * result_reason,
	// inputs
	__read_only image3d_t f1_image,
	__read_only image3d_t th1_image,
	__read_only image3d_t ph1_image,
	__read_only image3d_t f2_image,
	__read_only image3d_t th2_image,
	__read_only image3d_t ph2_image,
	//__global int *idx2cub,
	//__global int *seed_direction,
	__global float4 *start_coordinate,
	__global float4 *start_direction,
	int round,  //第几轮
	int iteration, //循环次数	
	// parameters
	float step_length,
	//float f_threshold,
	float angular_threshold, 
	//int count_threshold,
	
	int numValidVoxel,	
	__global float * debug,
	int dim_x,
	int dim_y,
	int dim_z,
	float reciprocal_pixdim_x,
	float reciprocal_pixdim_y,
	float reciprocal_pixdim_z
)
{
	
	if (get_global_id(0) < numValidVoxel)
	{
		int reason = 0;

		float4 coordinate_current = start_coordinate[get_global_id(0)];
		sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#ifdef USE_F_THRESHOLD_AS_TERMINATE_CONDITION
		float f_current;
#endif

		float4 valth, valph;
		float4 direction_current;
		float4 direction_next;
		float4 direction_compare;
		float4 direction_affine;
		
		float4 temp1;
		float4 temp2;
		float dist;
		__local float coordinatex[8 * LOCAL_SIZE];
		__local float coordinatey[8 * LOCAL_SIZE];
		__local float coordinatez[8 * LOCAL_SIZE];
				

#ifdef RECORD_INTEGER_POINT_ONLY
		float4 min_coordinate = (float4)(-1.0f, -1.0f, -1.0f, 0.0f);
		int previous_point = -1;
#endif
		int i = 0; 
		float total_distance;
		float angular_turn; 

		if (round == 0)
		{
			
			bool seed_direction = (get_global_id(0)<numValidVoxel);

			#ifdef USE_F_THRESHOLD_AS_TERMINATE_CONDITION
				valth = read_imagef(seed_direction ? f1_image : f2_image , sampler, coordinate_current);
				f_current = valth.x;
			#endif
			
			//valth = read_imagef((th2_image), sampler, coordinate_current);
			valth = seed_direction?read_imagef(th1_image, sampler, coordinate_current):read_imagef(th2_image, sampler, coordinate_current);
			valph = seed_direction?read_imagef(ph1_image, sampler, coordinate_current):read_imagef(ph2_image, sampler, coordinate_current);
			valth /= PI;
			valph /= PI;
			direction_next = (float4)(sinpi(valth.x) * cospi(valph.x), sinpi(valth.x) * sinpi(valph.x), cospi(valth.x), 0.0f);
			
		}
		else
		{
			direction_current = start_direction[get_global_id(0)];

			coordinatex[0 * get_local_size(0) + get_local_id(0)] = floor(coordinate_current.x);
			coordinatex[1 * get_local_size(0) + get_local_id(0)] = floor(coordinate_current.x);
			coordinatex[2 * get_local_size(0) + get_local_id(0)] = floor(coordinate_current.x);
			coordinatex[3 * get_local_size(0) + get_local_id(0)] = floor(coordinate_current.x);
			coordinatex[4 * get_local_size(0) + get_local_id(0)] = ceil(coordinate_current.x);
			coordinatex[5 * get_local_size(0) + get_local_id(0)] = ceil(coordinate_current.x);
			coordinatex[6 * get_local_size(0) + get_local_id(0)] = ceil(coordinate_current.x);
			coordinatex[7 * get_local_size(0) + get_local_id(0)] = ceil(coordinate_current.x);
			coordinatey[0 * get_local_size(0) + get_local_id(0)] = floor(coordinate_current.y);
			coordinatey[1 * get_local_size(0) + get_local_id(0)] = floor(coordinate_current.y);
			coordinatey[4 * get_local_size(0) + get_local_id(0)] = floor(coordinate_current.y);
			coordinatey[5 * get_local_size(0) + get_local_id(0)] = floor(coordinate_current.y);
			coordinatey[2 * get_local_size(0) + get_local_id(0)] = ceil(coordinate_current.y);
			coordinatey[3 * get_local_size(0) + get_local_id(0)] = ceil(coordinate_current.y);
			coordinatey[6 * get_local_size(0) + get_local_id(0)] = ceil(coordinate_current.y);
			coordinatey[7 * get_local_size(0) + get_local_id(0)] = ceil(coordinate_current.y);
			coordinatez[0 * get_local_size(0) + get_local_id(0)] = floor(coordinate_current.z);
			coordinatez[2 * get_local_size(0) + get_local_id(0)] = floor(coordinate_current.z);
			coordinatez[4 * get_local_size(0) + get_local_id(0)] = floor(coordinate_current.z);
			coordinatez[6 * get_local_size(0) + get_local_id(0)] = floor(coordinate_current.z);
			coordinatez[1 * get_local_size(0) + get_local_id(0)] = ceil(coordinate_current.z);
			coordinatez[3 * get_local_size(0) + get_local_id(0)] = ceil(coordinate_current.z);
			coordinatez[5 * get_local_size(0) + get_local_id(0)] = ceil(coordinate_current.z);
			coordinatez[7 * get_local_size(0) + get_local_id(0)] = ceil(coordinate_current.z);
				
			total_distance = 0;

			#ifdef USE_F_THRESHOLD_AS_TERMINATE_CONDITION
				f_current = 0;
			#endif
			
			direction_next = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
			/*direction_compare= (float4)(0.0f, 0.0f, 0.0f, 0.0f);
			float4 coordinate = (float4)(coordinatex[get_local_id(0)], coordinatey[get_local_id(0)], coordinatez[get_local_id(0)], 0.0f);
		    valth = read_imagef(f1_image, sampler, coordinate);
			valph = read_imagef(f2_image, sampler, coordinate);
			*/
			/*if(valth.x<f_threshold && valph.x<f_threshold)
			{
				reason=2;
				goto END;
			}
			else
			{*/
			for (int i = 0; i < 8; i++)
			{
			
				float4 coordinate = (float4)(coordinatex[i * get_local_size(0) + get_local_id(0)], coordinatey[i * get_local_size(0) + get_local_id(0)], coordinatez[i * get_local_size(0) + get_local_id(0)], 0.0f);
				dist = fast_distance(coordinate, coordinate_current);

				#ifdef USE_F_THRESHOLD_AS_TERMINATE_CONDITION
					total_distance += dist;
				#endif

				valth = read_imagef(th1_image, sampler, coordinate);
				valph = read_imagef(ph1_image, sampler, coordinate);
				valth /= PI;
				valph /= PI;
				temp1 = (float4)(sinpi(valth.x) * cospi(valph.x), sinpi(valth.x) * sinpi(valph.x), cospi(valth.x), 0.0f);
				float dot1 = dot(temp1, direction_current);
			
				valth = read_imagef(th2_image, sampler, coordinate);
				valph = read_imagef(ph2_image, sampler, coordinate);
				valth /= PI;
				valph /= PI;
				temp2 = (float4)(sinpi(valth.x) * cospi(valph.x), sinpi(valth.x) * sinpi(valph.x), cospi(valth.x), 0.0f);
				float dot2 = dot(temp2, direction_current);

				valth = read_imagef(f1_image, sampler, coordinate);
				valph = read_imagef(f2_image, sampler, coordinate);

				/*direction_compare=normalize(direction_next);
				angular_turn = dot(direction_compare, direction_current);
			    //if (angular_turn < angular_turn)
			    if (fabs(angular_turn) < angular_threshold)
			    {
				    reason = 4;
				    goto END;
			    }*/

				/*if(valth.x<f_threshold && valph.x<f_threshold)
				{
					reason=2;
					goto  END;
				}
				else if(valth.x<f_threshold)
				{
					temp1=temp2;
					dot1=dot2;
				}
				else if(valph.x<f_threshold)
				{
					temp1=temp1;
					dot1=dot1;
				}
				else
				{*/
				bool dir = (valth.x * fabs(dot1) > valph.x * fabs(dot2));    //diffusion direction 1 or 2
				valth.x = (dir) ? valth.x : valph.x;
				dot1 = (dir) ? dot1 : dot2;
				temp1 = (dir) ? temp1 : temp2;
				if (valth.x < 1e-6)
				{
					reason = 2;
					goto END;
				}
				//}
			
				(dot1>0) ? (direction_next += temp1 * dist) : (direction_next -= temp1 * dist);
				
				#ifdef USE_F_THRESHOLD_AS_TERMINATE_CONDITION
					f_current += valth.x * dist;
				#endif
			//}

			#ifdef USE_F_THRESHOLD_AS_TERMINATE_CONDITION
				f_current /= total_distance;
			#endif
			
			direction_next = normalize(direction_next);
			//angular_turn = (2 - (direction_next.x - direction_current.x) * (direction_next.x - direction_current.x) - (direction_next.y - direction_current.y) * (direction_next.y - direction_current.y) - (direction_next.z - direction_current.z) * (direction_next.z - direction_current.z)) / 2; 
			angular_turn = dot(direction_next, direction_current);
			//if (angular_turn < angular_turn)
			if (fabs(angular_turn) < angular_threshold)
			{
				reason = 4;
				goto END;
			}
			}
		}

		direction_current = direction_next;
		direction_affine = (float4)(direction_current.x * reciprocal_pixdim_x, direction_current.y * reciprocal_pixdim_y, direction_current.z * reciprocal_pixdim_z, 0.0f);
		direction_affine = normalize(direction_affine);

		coordinate_current += step_length * direction_affine;
		//coordinate_current += step_length * direction_current;
		if (coordinate_current.x<0 || coordinate_current.y<0 ||coordinate_current.z<0 || coordinate_current.x>=dim_x || coordinate_current.y>=dim_y || coordinate_current.z>=dim_z)
		{	
			reason = 2;
			goto END;
		}
		result[0 * numValidVoxel + get_global_id(0)] = coordinate_current;

		//if(get_global_id(0) == 31711)
		//{
		//	debug[0] = coordinate_current.x;
		//	debug[1] = coordinate_current.y;
		//	debug[2] = coordinate_current.z;
		//	debug[3] = direction_next.x;
		//	debug[4] = direction_next.y;
		//	debug[5] = direction_next.z;
		//	debug[6] = 0;
		//}

		
		for (i = 1; i < iteration; i++)
		{
			

			
			#ifdef RECORD_INTEGER_POINT_ONLY
			if ((int)(min_coordinate.x + min_coordinate.y * dim_x + min_coordinate.z * dim_x * dim_y) != previous_point)
			{
				previous_point = (int)(min_coordinate.x + min_coordinate.y * dim_x + min_coordinate.z * dim_x * dim_y);
				result[step * numValidVoxel + get_global_id(0)] = previous_point;
				step ++;
			}
			#endif
			
			coordinatex[0 * get_local_size(0) + get_local_id(0)] = floor(coordinate_current.x);
			coordinatex[1 * get_local_size(0) + get_local_id(0)] = floor(coordinate_current.x);
			coordinatex[2 * get_local_size(0) + get_local_id(0)] = floor(coordinate_current.x);
			coordinatex[3 * get_local_size(0) + get_local_id(0)] = floor(coordinate_current.x);
			coordinatex[4 * get_local_size(0) + get_local_id(0)] = ceil(coordinate_current.x);
			coordinatex[5 * get_local_size(0) + get_local_id(0)] = ceil(coordinate_current.x);
			coordinatex[6 * get_local_size(0) + get_local_id(0)] = ceil(coordinate_current.x);
			coordinatex[7 * get_local_size(0) + get_local_id(0)] = ceil(coordinate_current.x);
			coordinatey[0 * get_local_size(0) + get_local_id(0)] = floor(coordinate_current.y);
			coordinatey[1 * get_local_size(0) + get_local_id(0)] = floor(coordinate_current.y);
			coordinatey[4 * get_local_size(0) + get_local_id(0)] = floor(coordinate_current.y);
			coordinatey[5 * get_local_size(0) + get_local_id(0)] = floor(coordinate_current.y);
			coordinatey[2 * get_local_size(0) + get_local_id(0)] = ceil(coordinate_current.y);
			coordinatey[3 * get_local_size(0) + get_local_id(0)] = ceil(coordinate_current.y);
			coordinatey[6 * get_local_size(0) + get_local_id(0)] = ceil(coordinate_current.y);
			coordinatey[7 * get_local_size(0) + get_local_id(0)] = ceil(coordinate_current.y);
			coordinatez[0 * get_local_size(0) + get_local_id(0)] = floor(coordinate_current.z);
			coordinatez[2 * get_local_size(0) + get_local_id(0)] = floor(coordinate_current.z);
			coordinatez[4 * get_local_size(0) + get_local_id(0)] = floor(coordinate_current.z);
			coordinatez[6 * get_local_size(0) + get_local_id(0)] = floor(coordinate_current.z);
			coordinatez[1 * get_local_size(0) + get_local_id(0)] = ceil(coordinate_current.z);
			coordinatez[3 * get_local_size(0) + get_local_id(0)] = ceil(coordinate_current.z);
			coordinatez[5 * get_local_size(0) + get_local_id(0)] = ceil(coordinate_current.z);
			coordinatez[7 * get_local_size(0) + get_local_id(0)] = ceil(coordinate_current.z);
			float min_distance = 100.0f;
			total_distance = 0;

			#ifdef USE_F_THRESHOLD_AS_TERMINATE_CONDITION
			f_current = 0;
			#endif

			direction_next = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
			/*direction_compare= (float4)(0.0f, 0.0f, 0.0f, 0.0f);
			float4 coordinate = (float4)(coordinatex[get_local_id(0)], coordinatey[get_local_id(0)], coordinatez[get_local_id(0)], 0.0f);
		    valth = read_imagef(f1_image, sampler, coordinate);
			valph = read_imagef(f2_image, sampler, coordinate);
			*/
			/*if(valth.x<f_threshold && valph.x<f_threshold)
			{
				reason=2;
				goto END;
			}
			else
			{*/
			for (int i = 0; i < 8; i++)
			{
			
				float4 coordinate = (float4)(coordinatex[i * get_local_size(0) + get_local_id(0)], coordinatey[i * get_local_size(0) + get_local_id(0)], coordinatez[i * get_local_size(0) + get_local_id(0)], 0.0f);
				dist = fast_distance(coordinate, coordinate_current);
				total_distance += exp(dist);


		        valth = read_imagef(f1_image, sampler, coordinate);
			    valph = read_imagef(f2_image, sampler, coordinate);
				valth /= PI;
				valph /= PI;
				temp1 = (float4)(sinpi(valth.x) * cospi(valph.x), sinpi(valth.x) * sinpi(valph.x), cospi(valth.x), 0.0f);
				float dot1 = dot(temp1, direction_current);
			
				valth = read_imagef(th2_image, sampler, coordinate);
				valph = read_imagef(ph2_image, sampler, coordinate);
				valth /= PI;
				valph /= PI;
				temp2 = (float4)(sinpi(valth.x) * cospi(valph.x), sinpi(valth.x) * sinpi(valph.x), cospi(valth.x), 0.0f);
				float dot2 = dot(temp2, direction_current);

				valth = read_imagef(f1_image, sampler, coordinate);
				valph = read_imagef(f2_image, sampler, coordinate);

				/*direction_compare=normalize(direction_next);
				angular_turn = dot(direction_compare, direction_current);
			    //if (angular_turn < angular_turn)
			    if (fabs(angular_turn) < angular_threshold)
			    {
				    reason = 4;
				    goto END;
			    }*/	
				/*
				if(valth.x<f_threshold && valph.x<f_threshold)
				{
					reason=2;
					goto  END;
				}
				else if(valth.x<f_threshold)
				{
					temp1=temp2;
					dot1=dot2;
				}
				else if(valph.x<f_threshold)
				{
					temp1=temp1;
					dot1=dot1;
				}
				else
				{*/
				bool dir = (valth.x * fabs(dot1) > valph.x * fabs(dot2));    //diffusion direction 1 or 2
				valth.x = (dir) ? valth.x : valph.x;
				dot1 = (dir) ? dot1 : dot2;
				temp1 = (dir) ? temp1 : temp2;
				if (valth.x < 1e-6)
				{
					reason = 2;
					goto END;
				}
				//}
			
				(dot1>0) ? (direction_next += temp1 * dist) : (direction_next -= temp1 * dist);
			
				#ifdef USE_F_THRESHOLD_AS_TERMINATE_CONDITION
					f_current += valth.x * dist;
				#endif
			
			//}

			#ifdef USE_F_THRESHOLD_AS_TERMINATE_CONDITION
				if (f_current < f_threshold)
				{
					reason = 3;
					break;
				}
			#endif
			
			direction_next = normalize(direction_next);
			//angular_turn = (2 - (direction_next.x - direction_current.x) * (direction_next.x - direction_current.x) - (direction_next.y - direction_current.y) * (direction_next.y - direction_current.y) - (direction_next.z - direction_current.z) * (direction_next.z - direction_current.z)) / 2; 
			angular_turn = dot(direction_next, direction_current);
			//if (angular_turn < angular_turn)

			debug[get_global_id(0)] = angular_turn;

			if (fabs(angular_turn) < angular_threshold)
			{
				reason = 4;
				break;
			}
			}
			direction_current = direction_next;
			direction_affine = (float4)(direction_current.x*reciprocal_pixdim_x, direction_current.y * reciprocal_pixdim_y, direction_current.z*reciprocal_pixdim_z, 0.0f);
			direction_affine = normalize(direction_affine);
			coordinate_current += step_length * direction_affine;

			//coordinate_current += step_length * direction_current;
			if (coordinate_current.x<0 || coordinate_current.y<0 ||coordinate_current.z<0 || coordinate_current.x>=dim_x || coordinate_current.y>=dim_y || coordinate_current.z>=dim_z)
			{	
				reason = 2;
				goto END;
			}
			result[i * numValidVoxel + get_global_id(0)] = coordinate_current;

			//if (get_global_id(0) == 31711)//55727
			//{
			//	debug[i * 7] = coordinate_current.x;
			//	debug[i * 7 + 1] = coordinate_current.y;
			//	debug[i * 7 + 2] = coordinate_current.z;
			//	debug[i * 7 + 3] = direction_current.x;
			//	debug[i * 7 + 4] = direction_current.y;
			//	debug[i * 7 + 5] = direction_current.z;
			//	debug[i * 7 + 6] = 0;
			//}

		}
		start_coordinate[get_global_id(0)] = coordinate_current;
		start_direction[get_global_id(0)] = direction_current;
END:
		//result_step[get_global_id(0)] = step;
		result_reason[get_global_id(0)] = reason;
		result_iteration[get_global_id(0)] = i;
	}
}
