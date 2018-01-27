//#define USE_F_THRESHOLD_AS_TERMINATE_CONDITION
//#define RECORD_INTEGER_POINT_ONLY
#define PI 3.141592653589793f
#define LOCAL_SIZE 256

#define ROUND_MAX 13
#define Num_Probtrack_Sample 500
#define Fiber_Length_Min 20
#define threshold 0
#define threshold_prob 10


/*************************************************************************/
// FLIP_X, Y, Z is defined if the image need to be reversed 
// in the corresponding directions.
// make the macro definition of the USE_PROBMASK if a probability mask is used,
// the probability mask can be derived from results of FSL ProbtrackX or from this projects
// FINAL_ALL, FINAL_SEED_TRACK indicate whether the tractography start from all voxels
// or just some seed voxels.
/*************************************************************************/
//#define FLIP_X
//#define FLIP_Y
//#define FLIP_Z
//#define USE_PROBMASK 
//#define FINAL_ALL
#define FINAL_SEED_TRACK
#define RANDLEN 999

float para_threshold_angular = 0.7f;
float para_step_length = 0.4f;
const int min_step = 50;